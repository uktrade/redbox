from adminplus.sites import AdminSitePlus
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from magic_link import urls as magic_link_urls

from .redbox_core import views

admin.site = AdminSitePlus()
admin.autodiscover()

auth_urlpatterns = [
    path("magic_link/", include(magic_link_urls)),
    path("sign-in/", views.sign_in_view, name="sign-in"),
    path(
        "sign-in-link-sent/",
        views.sign_in_link_sent_view,
        name="sign-in-link-sent",
    ),
    path("signed-out/", views.signed_out_view, name="signed-out"),
    path("sign-up-page-1", views.Signup1.as_view(), name="sign-up-page-1"),
    path("sign-up-page-2", views.Signup2.as_view(), name="sign-up-page-2"),
    path("sign-up-page-3", views.Signup3.as_view(), name="sign-up-page-3"),
    path("sign-up-page-4", views.Signup4.as_view(), name="sign-up-page-4"),
    path("sign-up-page-5", views.Signup5.as_view(), name="sign-up-page-5"),
    path("sign-up-page-6", views.Signup6.as_view(), name="sign-up-page-6"),
    path("sign-up-page-7", views.Signup7.as_view(), name="sign-up-page-7"),
]

if settings.LOGIN_METHOD == "sso":
    auth_urlpatterns.append(path("auth/", include("authbroker_client.urls")))

info_urlpatterns = [
    path("privacy-notice/", views.info_views.privacy_notice_view, name="privacy-notice"),
    path(
        "accessibility-statement/",
        views.accessibility_statement_view,
        name="accessibility-statement",
    ),
    path("support/", views.support_view, name="support"),
]

document_urlpatterns = [
    path("documents/", views.DocumentView.as_view(), name="documents"),
    path("documents/<uuid:doc_id>/delete-document/", views.delete_document, name="delete-document"),
    path("documents/<uuid:doc_id>/title/", views.DocumentsTitleView.as_view(), name="document-titles"),
    path("documents/upload/", views.upload_document, name="document-upload"),
    path("upload/", views.UploadView.as_view(), name="upload"),
    path("remove-doc/<uuid:doc_id>", views.remove_doc_view, name="remove-doc"),
    path("remove-all-docs", views.remove_all_docs_view, name="remove-all-docs"),
    path("documents/your-documents/", views.YourDocuments.as_view(), name="your-documents-initial"),
    path("documents/your-documents/<uuid:active_chat_id>/", views.YourDocuments.as_view(), name="your-documents"),
]

chat_urlpatterns = [
    path("chats/<uuid:chat_id>/", views.ChatsView.as_view(), name="chats"),
    path("chats/", views.ChatsView.as_view(), name="chats"),
    path("chat/<uuid:chat_id>/title/", views.ChatsTitleView.as_view(), name="chat-titles"),
    path("citations/<uuid:message_id>/", views.CitationsView.as_view(), name="citations"),
    path("ratings/<uuid:message_id>/", views.RatingsView.as_view(), name="ratings"),
    path("chats/<uuid:chat_id>/update-chat-feedback", views.UpdateChatFeedback.as_view(), name="chat-feedback"),
    path("chats/<uuid:chat_id>/delete-chat/", views.DeleteChat.as_view(), name="delete-chat"),
    path("chats/recent-chats/", views.RecentChats.as_view(), name="recent-chats-initial"),
    path("chats/<uuid:active_chat_id>/recent-chats/", views.RecentChats.as_view(), name="recent-chats"),
    path("chats/chat-window/", views.ChatWindow.as_view(), name="chat-window-initial"),
    path("chats/<uuid:active_chat_id>/chat-window/", views.ChatWindow.as_view(), name="chat-window"),
]

notification_urlpatterns = [
    path("send-team-addition-email/", views.send_team_addition_email_view, name="send-team-addition-email"),
]

team_urlpatterns = [
    path("team/<uuid:team_id>/add-member-row/<uuid:user_id>/", views.add_team_member_row_view, name="add-member-row"),
    path("team/edit-member-row/<member_id>/", views.edit_team_member_row_view, name="edit-member-row"),
    path("team/delete-member/<member_id>/", views.delete_team_member_row_view, name="delete-member"),
    path("team/<uuid:team_id>/add-member/", views.add_team_member_view, name="add-member"),
    path("team/edit-member/<member_id>/", views.edit_team_member_view, name="edit-member"),
    path("team/<uuid:team_id>/delete-team/", views.delete_team_view, name="delete-team"),
    path("team/create-team/", views.create_team_view, name="create-team"),
]

admin_urlpatterns = [
    path("admin/report/", include("django_plotly_dash.urls")),
    path("admin/", admin.site.urls),
]

other_urlpatterns = [
    path("", views.homepage_view, name="homepage"),
    path("health/", views.health, name="health"),
    path("file-status/", views.file_status_api_view, name="file-status"),
    path("file-ingest-errors/", views.file_ingest_errors_view, name="file-ingest-errors"),
    path(
        "check-demographics/", views.CheckDemographicsView.as_view(), name="check-demographics"
    ),  # Can be removed once profile overlay is enabled
    path("demographics/", views.DemographicsView.as_view(), name="demographics"),
    path("update-demographics", views.UpdateDemographicsView.as_view(), name="update-demographics"),
    path(".well-known/security.txt", views.SecurityTxtRedirectView.as_view(), name="security.txt"),
    path("security", views.SecurityTxtRedirectView.as_view(), name="security"),
    path("sitemap/", views.misc_views.sitemap_view, name="sitemap"),
    path("faq/", views.faq_view, name="faq"),
    path("file-icon/<str:ext>/", views.file_icon_view, name="file-icon"),
    path("settings/", views.SettingsView.as_view(), name="settings"),
]


api_url_patterns = [
    path("api/v0/users/", views.user_view_pre_alpha, name="user-view"),
    path("api/v0/messages/", views.message_view_pre_alpha, name="message-view"),
    path("api/v0/aws-credentials", views.aws_credentials_api, name="aws-credentials"),
]

urlpatterns = (
    info_urlpatterns
    + other_urlpatterns
    + auth_urlpatterns
    + chat_urlpatterns
    + document_urlpatterns
    + notification_urlpatterns
    + team_urlpatterns
    + admin_urlpatterns
    + api_url_patterns
)

if settings.DEBUG:
    urlpatterns = urlpatterns + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
