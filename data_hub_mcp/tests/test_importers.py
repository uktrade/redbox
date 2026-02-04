import uuid

import importers
import pytest
from sa_models import AccountManagementObjective, Company, Interaction, InvestmentProject


class TestImporters:
    @pytest.mark.parametrize(
        ("db_object", "in_data", "mapping", "expected_error"),
        [
            # Company
            (Company(), {"id": str(uuid.uuid4())}, {}, None),  # basic
            (Company(), {"id": str(uuid.uuid4())}, {"id": "some_nonexistent_column"}, Exception),  # bad mapping
            # Interaction
            (Interaction(), {"company_id": str(uuid.uuid4())}, {}, None),
            # InvestmentProject
            (InvestmentProject(), {"uk_company_id": str(uuid.uuid4())}, {}, None),
            # AccountManagementObjective
            (AccountManagementObjective(), {"company_id": str(uuid.uuid4())}, {}, None),
        ],
    )
    def test_dict_to_db_obj(self, db_object, in_data, mapping, expected_error):
        target = importers.dict_to_db_obj

        if expected_error:
            with pytest.raises(expected_error):
                target(db_object, in_data, mapping)

        else:
            assert type(target(db_object, in_data, mapping)) is type(db_object)
