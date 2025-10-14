from redbox_app.redbox_core.dash_apps import report_app
from redbox_app.redbox_core.views.api_views import aws_credentials_api, message_view_pre_alpha, user_view_pre_alpha
from redbox_app.redbox_core.views.auth_views import sign_in_link_sent_view, sign_in_view, signed_out_view
from redbox_app.redbox_core.views.chat_views import (
    ChatsTitleView,
    ChatsView,
    ChatWindow,
    DeleteChat,
    RecentChats,
    UpdateChatFeedback,
)
from redbox_app.redbox_core.views.citation_views import CitationsView
from redbox_app.redbox_core.views.demographics_views import (
    CheckDemographicsView,
    DemographicsView,
    UpdateDemographicsView,
)
from redbox_app.redbox_core.views.document_views import (
    DocumentsTitleView,
    DocumentView,
    UploadView,
    YourDocuments,
    delete_document,
    remove_all_docs_view,
    remove_doc_view,
    upload_document,
)
from redbox_app.redbox_core.views.file_views import (
    file_icon_view,
    file_ingest_errors_view,
    file_status_api_view,
)
from redbox_app.redbox_core.views.info_views import accessibility_statement_view, privacy_notice_view, support_view
from redbox_app.redbox_core.views.misc_views import (
    SecurityTxtRedirectView,
    faq_view,
    health,
    homepage_view,
)
from redbox_app.redbox_core.views.notification_views import send_team_addition_email_view
from redbox_app.redbox_core.views.ratings_views import RatingsView
from redbox_app.redbox_core.views.settings_views import SettingsView
from redbox_app.redbox_core.views.signup_views import Signup1, Signup2, Signup3, Signup4, Signup5, Signup6, Signup7
from redbox_app.redbox_core.views.team_views import (
    add_team_member_row_view,
    add_team_member_view,
    delete_team_member_row_view,
    delete_team_view,
    edit_team_member_row_view,
    edit_team_member_view,
)

__all__ = [
    "ChatsTitleView",
    "ChatsView",
    "CitationsView",
    "CheckDemographicsView",
    "DemographicsView",
    "DocumentView",
    "RatingsView",
    "SecurityTxtRedirectView",
    "UploadView",
    "YourDocuments",
    "upload_document",
    "UpdateDemographicsView",
    "file_status_api_view",
    "file_ingest_errors_view",
    "health",
    "homepage_view",
    "file_icon_view",
    "remove_doc_view",
    "remove_all_docs_view",
    "DocumentsTitleView",
    "delete_document",
    "privacy_notice_view",
    "accessibility_statement_view",
    "support_view",
    "sign_in_view",
    "sign_in_link_sent_view",
    "signed_out_view",
    "Signup1",
    "Signup2",
    "Signup3",
    "Signup4",
    "Signup5",
    "Signup6",
    "Signup7",
    "report_app",
    "UpdateChatFeedback",
    "DeleteChat",
    "RecentChats",
    "ChatWindow",
    "user_view_pre_alpha",
    "message_view_pre_alpha",
    "aws_credentials_api",
    "faq_view",
    "send_team_addition_email_view",
    "SettingsView",
    "add_team_member_row_view",
    "edit_team_member_row_view",
    "delete_team_member_row_view",
    "add_team_member_view",
    "edit_team_member_view",
    "delete_team_view",
]
