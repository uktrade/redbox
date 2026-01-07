import csv
import re
from datetime import datetime, timedelta

from sa_models import AccountManagementObjectives, Company, CompanyInteraction, InvestmentProjects


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
            elif repr(column.type) == "ARRAY(String())":
                strings = re.findall("(.*?)", value)
                tmp_value = []
                for s in strings:
                    tmp_value.append(s)

                value = tmp_value
            elif repr(column.type) == "ARRAY(UUID())":
                uuids = re.findall("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", value)
                tmp_value = []
                for uuid in uuids:
                    tmp_value.append(uuid)

                value = tmp_value
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
    # get_data_from_s3

    # TODO: check modified on date, only update when modified in the last week

    with open("data/company_data.csv") as infile:
        csv_reader = csv.DictReader(infile)
        data = [row for row in csv_reader]

        counter = 0

        for row in data:
            company_new = Company()
            company_new = dict_to_db_obj(company_new, row)

            # if company has been created or updated in the last 3 days, update
            days_ago = datetime.now() + timedelta(days=-import_newer_than_in_days)

            process_import = False

            if (company_new.modified_on is not None and company_new.modified_on > days_ago.date()) or (
                company_new.created_on is not None and company_new.created_on > days_ago.date()
            ):
                process_import = True

            if process_import:
                company_existing = session.query(Company).filter(Company.id == row["id"]).first()
                if company_existing:
                    company_existing = dict_to_db_obj(company_existing, row)
                    session.add(company_existing)
                else:
                    session.add(company_new)
                counter += 1

            if counter % 1000 == 0 and counter > 0:
                print(f"Committing company to DB: {counter}")
                session.commit()
            counter += 1

        session.commit()


def import_interactions(session, import_newer_than_in_days=3):
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
    session.query(CompanyInteraction).delete()
    session.commit()

    with open("data/company_interaction_data.csv") as infile:
        csv_reader = csv.DictReader(infile)
        data = [row for row in csv_reader]

        counter = 0
        row_counter = 0
        for row in data:
            # New object
            company_interaction = CompanyInteraction()
            company_interaction = dict_to_db_obj(company_interaction, row, mapping)

            days_ago = datetime.now() + timedelta(days=-import_newer_than_in_days)
            if company_interaction.date_of_interaction > days_ago.date():
                # check company exists
                company = session.query(Company).filter(Company.id == company_interaction.company_id).first()
                if company:
                    session.add(company_interaction)
                    counter += 1

            if counter % 1000 == 0 and counter > 0:
                print(f"Committing interaction to DB: {counter}")
                session.commit()

            if row_counter % 1000 == 0 and row_counter > 0:
                print(f"row counter interaction: {row_counter}")

            row_counter += 1

        session.commit()


def import_objectives(session, import_newer_than_in_days=3):
    mapping = {}

    # Remove all interactions
    session.query(CompanyInteraction).delete()
    session.commit()

    with open("data/objectives.csv") as infile:
        csv_reader = csv.DictReader(infile)
        data = [row for row in csv_reader]

        counter = 0
        row_counter = 0
        for row in data:
            print(row)
            # New object
            objective = AccountManagementObjectives()
            objective = dict_to_db_obj(objective, row, mapping)

            days_ago = datetime.now() + timedelta(days=-import_newer_than_in_days)
            if objective.date_of_interaction > days_ago.date():
                # check company exists
                company = session.query(Company).filter(Company.id == objective.company_id).first()
                if company:
                    session.add(objective)
                    counter += 1

            if counter % 1000 == 0 and counter > 0:
                print(f"Committing interaction to DB: {counter}")
                session.commit()

            if row_counter % 1000 == 0 and row_counter > 0:
                print(f"row counter interaction: {row_counter}")

            row_counter += 1

        session.commit()


def import_projects(session, import_newer_than_in_days=3):
    mapping = {}

    # Remove all interactions
    session.query(CompanyInteraction).delete()
    session.commit()

    with open("data/investment_projects.csv") as infile:
        csv_reader = csv.DictReader(infile)
        data = [row for row in csv_reader]

        counter = 0
        row_counter = 0
        for index, row in enumerate(data):
            # New object
            investment_project = InvestmentProjects()
            investment_project = dict_to_db_obj(investment_project, row, mapping)

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
                    # company = session.query(Company).filter(Company.id == investment_project.company_id).first()
                    session.add(investment_project)
                    session.commit()

        session.commit()
