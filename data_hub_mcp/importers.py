import csv
import re
from datetime import datetime, timedelta
import ast
import log

from sa_models import AccountManagementObjectives, Company, Interaction, InvestmentProjects


class ColumnNotFound(Exception):
    pass


class DateParseFailure(Exception):
    pass


class DateTimeParseFailure(Exception):
    pass


def dict_to_db_obj(db_obj, data_dict, mapping: dict | None = None):
    for key, value in data_dict.items():
        updated_key = key
        if mapping and key in mapping.keys():
            updated_key = mapping[key]

        column = None
        for col in db_obj.__table__.columns:
            if col.name == updated_key:
                column = col
                break

        if column is None:
            raise ColumnNotFound(f"Cant find matching column for {updated_key}")

        if value == "":
            value = None

        if str(column.type) == "BOOLEAN":
            if value == "":
                value = None
            else:
                value = bool(value)

        if str(column.type) == "DATE":
            if value == "" or value is None:
                value = None
            elif re.search("^[0-9]{2}/[0-9]{2}/[0-9]{4}$", value):
                value = datetime.strptime(value, "%d/%m/%Y").date()
            elif re.search("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", value):
                value = datetime.strptime(value, "%Y-%m-%d").date()
            elif re.search("^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$", value):
                value = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").date()
            else:
                raise DateParseFailure(f"Cant parse date for {value}")

        if str(column.type) == "DATETIME":
            # '2025-10-07 23:09:42.394611'
            if value is not None:
                if re.search("^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}[.][0-9]{6}$", value):
                    value = datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")
                elif re.search("^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$", value):
                    value = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                elif re.search("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", value):
                    value = datetime.strptime(value, "%Y-%m-%d")
                else:
                    raise DateTimeParseFailure(f"Cant parse datetime for {value}")

        if str(column.type) == "ARRAY":
            if value == "[None]" or value is None:
                value = []
            elif repr(column.type) in ["ARRAY(String())", "ARRAY(String(length=100))", "ARRAY(UUID())"]:
                value = ast.literal_eval(value)
            else:
                value = [value]

        if str(column.type) == "INTEGER" or str(column.type) == "BIGINT":
            if value == "" or value is None:
                value = None
            else:
                value = int(round(float(value), 0))

        db_obj.__setattr__(updated_key, value)

    return db_obj


def import_companies(session, import_newer_than_in_days=3):
    log.logger.info('Begin importing companies')
    with open("data/companies.csv") as infile:
        csv_reader = csv.DictReader(infile)
        data = [row for row in csv_reader]

        counter = 0

        for row in data:
            company_new = Company()
            company_new = dict_to_db_obj(company_new, row)

            # if company has been created or updated in the last 3 days, update
            days_ago = datetime.now() + timedelta(days=-import_newer_than_in_days)

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
                    log.logger.info('updating company')
                else:
                    session.add(company_new)
                    log.logger.info('adding company')
                counter += 1

            if counter % 1000 == 0 and counter > 0:
                log.logger.info(f"Committing company to DB: {counter}")
                session.commit()
            counter += 1

        session.commit()
        log.logger.info('End importing companies')


def import_interactions(session, import_newer_than_in_days=3):
    log.logger.info('Begin importing interactions')
    mapping = {
        "Interaction Year": "interaction_year",
        "Interaction Financial Year": "interaction_financial_year",
        "Company Name": "company_name",
        "Company Sector": "company_sector",
        "Company ID": "company_id",
        "Date of Interaction": "date_of_interaction",
        "Interaction Subject": "interaction_subject",
        "Interaction Theme: Investment or Export": "interaction_theme_investment_or_export",
        "Adviser Name": "adviser_name",
        "Team Name": "team_name",
    }

    # Remove all interactions
    session.query(Interaction).delete()
    session.commit()

    with open("data/company_interactions.csv") as infile:
        csv_reader = csv.DictReader(infile)
        data = [row for row in csv_reader]

        counter = 0
        for row in data:
            # New object
            company_interaction = Interaction()
            company_interaction = dict_to_db_obj(company_interaction, row, mapping)
            days_ago = datetime.now() + timedelta(days=-import_newer_than_in_days)
            if company_interaction.date_of_interaction > days_ago.date():
                # check company exists
                company = session.query(Company).filter(Company.id == company_interaction.company_id).first()
                if company:
                    session.add(company_interaction)
                    log.logger.info('adding interaction')
                    counter += 1
                else:
                    log.logger.info(f'No match for interaction company {company_interaction.company_id}')

            if counter % 1000 == 0 and counter > 0:
                session.commit()

        session.commit()

        log.logger.info('End importing interactions')


def import_objectives(session, import_newer_than_in_days=3):
    log.logger.info('Begin importing objectives')
    mapping = {}

    with open("data/objectives.csv") as infile:
        csv_reader = csv.DictReader(infile)
        data = [row for row in csv_reader]

        counter = 0
        for row in data:
            objective = AccountManagementObjectives()
            objective = dict_to_db_obj(objective, row, mapping)

            objective_existing = session.query(AccountManagementObjectives).filter(AccountManagementObjectives.id == row["id"]).first()

            if objective_existing:
                objective = dict_to_db_obj(objective_existing, row)

            days_ago = datetime.now() + timedelta(days=-import_newer_than_in_days)
            if objective.modified_on.date() > days_ago.date():
                # check company exists
                company = session.query(Company).filter(Company.id == objective.company_id).first()
                if company:
                    session.add(objective)
                    log.logger.info('adding / updating objective')
                    counter += 1
                else:
                    log.logger.info(f'No match for objective company {objective.company_id}')

        session.commit()
        log.logger.info('End importing objectives')


def import_projects(session, import_newer_than_in_days=3):
    log.logger.info('Begin importing projects')
    mapping = {}

    with open("data/investment_projects.csv") as infile:
        csv_reader = csv.DictReader(infile)
        data = [row for row in csv_reader]

        for index, row in enumerate(data):
            # New object
            investment_project = InvestmentProjects()
            investment_project = dict_to_db_obj(investment_project, row, mapping)

            investment_project_existing = session.query(InvestmentProjects).filter(InvestmentProjects.id == row["id"]).first()

            if investment_project_existing:
                investment_project = dict_to_db_obj(investment_project_existing, row)

            days_ago = datetime.now() + timedelta(days=-import_newer_than_in_days)  # noqa: DTZ005

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
                    log.logger.info('adding / updating investment project')
                    session.commit()
                else:
                    log.logger.info(f'No match for project company {investment_project.uk_company_id}')

    log.logger.info('End importing projects')
