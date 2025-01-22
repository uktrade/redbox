from django.contrib.auth import get_user_model
from rest_framework import serializers

from redbox_app.redbox_core.models import Chat, ChatMessage, ChatMessageTokenUse, File

User = get_user_model()


class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = File
        fields = ("file_name", "created_at")


class ChatMessageTokenUseSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessageTokenUse
        fields = ("use_type", "model_name", "token_count")


class ChatMessageSerializer(serializers.ModelSerializer):
    selected_files = FileSerializer(many=True, read_only=True)
    source_files = FileSerializer(many=True, read_only=True)
    token_use = ChatMessageTokenUseSerializer(
        source="chatmessagetokenuse_set", many=True, read_only=True
    )

    class Meta:
        model = ChatMessage
        fields = (
            "id",
            "created_at",
            "text",
            "role",
            "route",
            "selected_files",
            "source_files",
            "rating",
            "rating_text",
            "rating_chips",
            "token_use",
        )


class ChatSerializer(serializers.ModelSerializer):
    messages = ChatMessageSerializer(
        source="chatmessage_set", many=True, read_only=True
    )

    class Meta:
        model = Chat
        fields = ("name", "messages", "id")


class UserSerializer(serializers.ModelSerializer):
    chats = ChatSerializer(source="chat_set", many=True, read_only=True)

    class Meta:
        model = User
        fields = (
            "ai_experience",
            "business_unit",
            "grade",
            "profession",
            "is_staff",
            "is_developer",
            "role",
            "accessibility_options",
            "accessibility_categories",
            "accessibility_description",
            "digital_confidence",
            "usage_at_work",
            "usage_outside_work",
            "how_useful",
            "task_1_description",
            "task_1_regularity",
            "task_1_duration",
            "task_1_consider_using_ai",
            "task_2_description",
            "task_2_regularity",
            "task_2_duration",
            "task_2_consider_using_ai",
            "task_3_description",
            "task_3_regularity",
            "task_3_duration",
            "task_3_consider_using_ai",
            "role_regularity_summarise_large_docs",
            "role_regularity_condense_multiple_docs",
            "role_regularity_search_across_docs",
            "role_regularity_compare_multiple_docs",
            "role_regularity_specific_template",
            "role_regularity_shorten_docs",
            "role_regularity_write_docs",
            "role_duration_summarise_large_docs",
            "role_duration_condense_multiple_docs",
            "role_duration_search_across_docs",
            "role_duration_compare_multiple_docs",
            "role_duration_specific_template",
            "role_duration_shorten_docs",
            "role_duration_write_docs",
            "consent_research",
            "chats",
        )
