from redbox_app.redbox_core.services import chats as chat_service


class ChatContextMixin:
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context.update(chat_service.get_context(self.request))

        return context
