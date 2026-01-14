import sa_models
from db_ops import get_engine, get_session
from importers import (
    import_objectives,
    import_projects, import_companies, import_interactions,
)
from dotenv import load_dotenv

load_dotenv()

engine = get_engine()

metadata_obj = sa_models.Base.metadata

# metadata_obj.drop_all(engine)
metadata_obj.create_all(engine)

session = get_session(engine)
import_companies(session, 30000)
session.close()

session = get_session(engine)
import_interactions(session, 30000)
session.close()

session = get_session(engine)
import_projects(session, 30000)
session.close()

session = get_session(engine)
import_objectives(session, 30000)
session.close()

