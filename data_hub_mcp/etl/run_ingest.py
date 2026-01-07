import sa_models
from data_hub_company_and_interaction_data.importers import (
    import_objectives,
    import_projects,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

engine = create_engine("postgresql+psycopg2://dougmills:123123@localhost/data_hub_data")

metadata_obj = sa_models.Base.metadata

# metadata_obj.drop_all(engine)
metadata_obj.create_all(engine)

session = Session(engine)

# import_companies(session, 30)
# import_interactions(session, 30)
import_projects(session, 30)
import_objectives(session, 30)
