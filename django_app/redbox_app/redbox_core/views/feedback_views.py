from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from waffle import flag_is_active

from redbox_app.redbox_core.forms import ChatMessageFeedbackForm
from redbox_app.redbox_core.models import ChatMessage, ChatMessageFeedback

FEEDBACK_FLAG = "enable_feedback_redesign"
FORM_TEMPLATE = "chat/message/feedback/_feedback-form.html"
BUTTONS_TEMPLATE = "chat/message/feedback/_feedback_buttons.html"
THANKS_TEMPLATE = "chat/message/feedback/_feedback-thanks.html"


@login_required
@require_http_methods(["GET"])
def get_feedback_buttons(request, message_id):
    if not flag_is_active(request, FEEDBACK_FLAG):
        raise Http404

    message = get_object_or_404(
        ChatMessage.objects.filter(chat__user=request.user),
        id=message_id,
    )

    instance = ChatMessageFeedback.objects.filter(message=message).first()
    context = {"message_id": message.id, "message": message}

    if instance is None:
        return render(request, BUTTONS_TEMPLATE, context)

    return render(request, THANKS_TEMPLATE, context)


@login_required
@require_http_methods(["POST", "DELETE"])
def chat_message_feedback(request, message_id):
    if not flag_is_active(request, FEEDBACK_FLAG):
        raise Http404

    message = get_object_or_404(
        ChatMessage.objects.filter(chat__user=request.user),
        id=message_id,
    )

    instance = ChatMessageFeedback.objects.filter(message=message).first()
    context = {"message_id": message.id, "message": message}

    if request.method == "DELETE":
        if instance is not None:
            instance.delete()
        return render(request, BUTTONS_TEMPLATE, context)

    form = ChatMessageFeedbackForm(request.POST, instance=instance)
    if not form.is_valid():
        response = render(request, FORM_TEMPLATE, {**context, "form": form}, status=422)
        response["HX-Reswap"] = "innerHTML"
        return response

    feedback, _ = ChatMessageFeedback.objects.update_or_create(
        message=message,
        defaults={
            "is_positive": form.cleaned_data["is_positive"],
            "reason": form.cleaned_data["reason"],
            "detail": form.cleaned_data["detail"],
        },
    )

    show_form = request.GET.get("show_form") == "true"
    if show_form:
        form = ChatMessageFeedbackForm(instance=feedback)
        return render(request, FORM_TEMPLATE, {**context, "form": form})

    return redirect(reverse("chat-message-feedback-buttons", kwargs={"message_id": message_id}))
