from datetime import date

from django import forms
from django.http import HttpResponse
from django.template.loader import render_to_string
from django.utils import timezone

from redbox_app.redbox_core.types import RenderTemplateItem


def get_date_group(on: date) -> str:
    today = timezone.now().date()
    if on == today:
        return "Today"
    return "Previous"


def render_with_oob(templates: list[RenderTemplateItem]) -> HttpResponse:
    """
    Render multiple templates with their own context, request and optional engine.
    Using HTMX Out of bounds swap method.

    Args:
        templates (List[RenderTemplateItem]): A list of dicts like:
            {
                "template": str,
                "context": dict,
                "request": HttpRequest,
                "using": Optional[str]
            }
    Returns:
        HttpResponse: All rendered templates concatenated into a single response.
    """

    html = ""
    for template_item in templates:
        template = template_item["template"]
        context = template_item["context"]
        request = template_item["request"]
        engine = template_item.get("engine", "jinja2")  # Default to jinja2
        html += render_to_string(template, context, request, using=engine)

    return HttpResponse(html)


def save_forms(form_dict: forms.BaseForm | dict[str, forms.BaseForm]):
    results = {}

    if isinstance(form_dict, forms.BaseForm):
        results["form"] = form_dict.is_valid()
        if results["form"]:
            form_dict.save()
    else:
        for name, form in form_dict.items():
            results[name] = form.is_valid()
            if results[name]:
                form.save()

    return results
