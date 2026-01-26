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
    RefreshFragmentsView,
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
    create_team_view,
    delete_team_member_row_view,
    delete_team_view,
    edit_team_member_row_view,
    edit_team_member_view,
)
from redbox_app.redbox_core.views.tools_views import ToolsView, tool_info_page_view

__all__ = [
    "ChatWindow",
    "ChatsTitleView",
    "ChatsView",
    "CheckDemographicsView",
    "CitationsView",
    "DeleteChat",
    "DemographicsView",
    "DocumentView",
    "DocumentsTitleView",
    "RatingsView",
    "RecentChats",
    "RefreshFragmentsView",
    "SecurityTxtRedirectView",
    "SettingsView",
    "Signup1",
    "Signup2",
    "Signup3",
    "Signup4",
    "Signup5",
    "Signup6",
    "Signup7",
    "ToolsView",
    "UpdateChatFeedback",
    "UpdateDemographicsView",
    "UploadView",
    "YourDocuments",
    "accessibility_statement_view",
    "add_team_member_row_view",
    "add_team_member_view",
    "aws_credentials_api",
    "create_team_view",
    "delete_document",
    "delete_team_member_row_view",
    "delete_team_view",
    "edit_team_member_row_view",
    "edit_team_member_view",
    "faq_view",
    "file_icon_view",
    "file_ingest_errors_view",
    "file_status_api_view",
    "health",
    "homepage_view",
    "message_view_pre_alpha",
    "privacy_notice_view",
    "remove_all_docs_view",
    "remove_doc_view",
    "report_app",
    "send_team_addition_email_view",
    "sign_in_link_sent_view",
    "sign_in_view",
    "signed_out_view",
    "support_view",
    "tool_info_page_view",
    "upload_document",
    "user_view_pre_alpha",
]
