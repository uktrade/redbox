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
    file_ingest_errors,
    file_status_api_view,
    remove_all_docs_view,
    remove_doc_view,
    upload_document,
)
from redbox_app.redbox_core.views.info_views import accessibility_statement_view, privacy_notice_view, support_view
from redbox_app.redbox_core.views.misc_views import SecurityTxtRedirectView, faq_view, health, homepage_view
from redbox_app.redbox_core.views.ratings_views import RatingsView
from redbox_app.redbox_core.views.signup_views import Signup1, Signup2, Signup3, Signup4, Signup5, Signup6, Signup7

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
    "file_ingest_errors",
    "health",
    "homepage_view",
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
]
