from django.apps import apps
from django.test import override_settings


class TestApps:
    @override_settings(ENVIRONMENT="INTEGRATION")
    def test_app_name_is_set_correctly(self):
        assert apps.get_app_config("redbox_core").name == "redbox_app.redbox_core"
