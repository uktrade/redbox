from django.template.response import TemplateResponse

from redbox_app.redbox_core.services import chats as chat_service


def render_your_documents(request, active_chat_id) -> TemplateResponse:
    context = chat_service.get_context(request, active_chat_id)

    return TemplateResponse(
        request,
        "side_panel/your_documents_list.html",
        context,
    )
