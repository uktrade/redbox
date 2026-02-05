import ast
import csv
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import log
from sa_models import AccountManagementObjective, Company, Interaction, InvestmentProject


class ColumnNotFoundError(Exception):
    pass


class ColumnTypeParserNotFoundError(Exception):
    pass


class DateParseError(Exception):
    pass


class DateTimeParseError(Exception):
    pass


def dict_to_db_obj(db_obj, data_dict, mapping: dict | None = None):  # noqa: C901
    for key, value in data_dict.items():
        updated_key = key
        if mapping and key in mapping:
            updated_key = mapping[key]

        column = find_column(db_obj, updated_key)

        if column is None:
            err_msg = f"Cant find matching column for {updated_key}"
            raise ColumnNotFoundError(err_msg)

        formatted_value = None

        if str(column.type) == "BOOLEAN":
            formatted_value = bool(value)

        elif str(column.type) == "DATE":
            formatted_value = parse_date_string(value)

        elif str(column.type) == "DATETIME":
            formatted_value = parse_datetime_string(value)

        elif str(column.type) == "ARRAY":
            formatted_value = parse_array_string(column, value)

        elif str(column.type) == "INTEGER" or str(column.type) == "BIGINT":
            formatted_value = parse_integer_string(formatted_value, value)

        elif str(column.type) == "FLOAT":
            formatted_value = parse_float_string(formatted_value, value)

        elif str(column.type).startswith("VARCHAR") or str(column.type) == "UUID":
            formatted_value = value

        else:
            err_msg = f"Cant parse colum type {column.type!s}"
            raise ColumnTypeParserNotFoundError(err_msg)

        if formatted_value == "":
            formatted_value = None

        db_obj.__setattr__(updated_key, formatted_value)

    return db_obj


def parse_float_string(formatted_value, value) -> float:
    if value != "":
        formatted_value = float(value)

    return formatted_value


def find_column(db_obj, updated_key) -> Any:
    column = None
    for col in db_obj.__table__.columns:
        if col.name == updated_key:
            column = col
            break
    return column


def parse_integer_string(formatted_value, value: Literal[""] | None | Any) -> Any:
    if value != "" and value is not None:
        formatted_value = int(round(float(value), 0))

    return formatted_value


def parse_array_string(column: Any | None, value: Literal[""] | None | Any) -> list[Any]:
    if value in ["[None]", "", None]:
        formatted_value = []
    elif repr(column.type) in ["ARRAY(String())", "ARRAY(String(length=100))", "ARRAY(UUID())"]:
        formatted_value = ast.literal_eval(value)
    else:
        formatted_value = [value]

    return formatted_value


def parse_datetime_string(value: Literal[""] | Any) -> datetime | None:
    if value == "" or value is None:
        return None

    # '2025-10-07 23:09:42.394611'
    if re.search("^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}[.][0-9]{6}$", value):
        formatted_value = datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=UTC)
    elif re.search("^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[.][0-9]{6}$", value):
        formatted_value = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=UTC)
    elif re.search("^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$", value):
        formatted_value = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    elif re.search("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", value):
        formatted_value = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        err_msg = f"Cant parse datetime for {value}"
        raise DateTimeParseError(err_msg)
    return formatted_value


def parse_date_string(value: Literal[""] | Any) -> Any:
    if value == "" or value is None:
        formatted_value = None
    elif re.search("^[0-9]{2}/[0-9]{2}/[0-9]{4}$", value):
        formatted_value = datetime.strptime(value, "%d/%m/%Y").replace(tzinfo=UTC).date()
    elif re.search("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", value):
        formatted_value = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC).date()
    elif re.search("^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$", value):
        formatted_value = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).date()
    else:
        err_msg = f"Cant parse date for {value}"
        raise DateParseError(err_msg)
    return formatted_value


def import_companies(session, import_newer_than_in_days=3):
    log.logger.info("Begin importing companies")
    with Path("data/companies.csv").open() as infile:
        csv_reader = csv.DictReader(infile)
        data = list(csv_reader)

        counter = 0

        for row in data:
            company_new = Company()
            company_new = dict_to_db_obj(company_new, row)
            log.logger.info(f"company data : {company_new!s}")
            days_ago = datetime.now(tz=UTC) + timedelta(days=-import_newer_than_in_days)

            process_import = False

            if (company_new.modified_on is not None and company_new.modified_on.date() > days_ago.date()) or (
                company_new.created_on is not None and company_new.created_on > days_ago.date()
            ):
                process_import = True

            if process_import:
                company_existing = session.query(Company).filter(Company.id == row["id"]).first()
                if company_existing:
                    company_existing = dict_to_db_obj(company_existing, row)
                    session.add(company_existing)
                    log.logger.info("updating company")
                else:
                    session.add(company_new)
                    log.logger.info("adding company")
                counter += 1

            if counter % 1000 == 0 and counter > 0:
                log.logger.info(f"Committing company to DB: {counter}")
                session.commit()
            counter += 1

        session.commit()
        log.logger.info("End importing companies")


def import_interactions(session, import_newer_than_in_days=3):
    log.logger.info("Begin importing interactions")

    # Remove all interactions
    session.query(Interaction).delete()
    session.commit()

    with Path("data/company_interactions.csv").open() as infile:
        csv_reader = csv.DictReader(infile)
        data = list(csv_reader)

        counter = 0
        for row in data:
            company_interaction = dict_to_db_obj(Interaction(), row)
            days_ago = datetime.now(tz=UTC) + timedelta(days=-import_newer_than_in_days)
            if company_interaction.interaction_date > days_ago.date():
                # check company exists
                company = session.query(Company).filter(Company.id == company_interaction.company_id).first()
                if company_interaction.investment_project_id is not None:
                    investment_project = (
                        session.query(InvestmentProject)
                        .filter(InvestmentProject.id == company_interaction.investment_project_id)
                        .first()
                    )
                    # If we cant match to the project make it None - could be an old project that s not imported
                    if not investment_project:
                        company_interaction.investment_project_id = None

                if company:
                    # Check interaction existing
                    company_interaction_existing = (
                        session.query(Interaction).filter(Interaction.id == row["id"]).first()
                    )

                    if company_interaction_existing:
                        company_interaction_existing = dict_to_db_obj(company_interaction_existing, row)
                        session.add(company_interaction_existing)
                        log.logger.info("updating company")
                    else:
                        session.add(company_interaction)
                        log.logger.info("adding company")

                else:
                    log.logger.info(f"No match for interaction company {company_interaction.company_id}")

                if company:
                    session.add(company_interaction)
                    log.logger.info("adding interaction")
                    counter += 1

            if counter % 1000 == 0 and counter > 0:
                session.commit()

        session.commit()

        log.logger.info("End importing interactions")


def import_objectives(session, import_newer_than_in_days=3):
    log.logger.info("Begin importing objectives")
    mapping = {}

    with Path("data/objectives.csv").open() as infile:
        csv_reader = csv.DictReader(infile)
        data = list(csv_reader)

        counter = 0
        for row in data:
            objective = AccountManagementObjective()
            objective = dict_to_db_obj(objective, row, mapping)

            objective_existing = (
                session.query(AccountManagementObjective).filter(AccountManagementObjective.id == row["id"]).first()
            )

            if objective_existing:
                objective = dict_to_db_obj(objective_existing, row)

            days_ago = datetime.now(tz=UTC) + timedelta(days=-import_newer_than_in_days)
            if objective.modified_on.date() > days_ago.date():
                # check company exists
                company = session.query(Company).filter(Company.id == objective.company_id).first()
                if company:
                    session.add(objective)
                    log.logger.info("adding / updating objective")
                    counter += 1
                else:
                    log.logger.info(f"No match for objective company {objective.company_id}")

        session.commit()
        log.logger.info("End importing objectives")


def import_projects(session, import_newer_than_in_days=3):
    log.logger.info("Begin importing projects")
    mapping = {}

    with Path("data/investment_projects.csv").open() as infile:
        csv_reader = csv.DictReader(infile)
        data = list(csv_reader)

        for row in data:
            # New object
            investment_project = InvestmentProject()
            investment_project = dict_to_db_obj(investment_project, row, mapping)

            investment_project_existing = (
                session.query(InvestmentProject).filter(InvestmentProject.id == row["id"]).first()
            )

            if investment_project_existing:
                investment_project = dict_to_db_obj(investment_project_existing, row)

            days_ago = datetime.now(tz=UTC) + timedelta(days=-import_newer_than_in_days)

            process_import = False

            if (
                investment_project.modified_on is not None and investment_project.modified_on.date() > days_ago.date()
            ) or (investment_project.created_on is not None and investment_project.created_on.date() > days_ago.date()):
                process_import = True

            if process_import:
                # check company exists
                uk_company = session.query(Company).filter(Company.id == investment_project.uk_company_id).first()
                if uk_company:
                    session.add(investment_project)
                    log.logger.info("adding / updating investment project")
                    session.commit()
                else:
                    log.logger.info(f"No match for project company {investment_project.uk_company_id}")

    log.logger.info("End importing projects")
