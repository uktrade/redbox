from django.contrib.auth import get_user_model
from rest_framework import serializers

from redbox_app.redbox_core.models import ChatMessage, ChatMessageTokenUse, File

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
    token_use = ChatMessageTokenUseSerializer(source="chatmessagetokenuse_set", many=True, read_only=True)
    chat_id = serializers.PrimaryKeyRelatedField(source="chat", read_only=True)
    user_id = serializers.PrimaryKeyRelatedField(source="chat.user", read_only=True)

    class Meta:
        model = ChatMessage
        fields = (
            "id",
            "created_at",
            "modified_at",
            "text",
            "role",
            "route",
            "selected_files",
            "source_files",
            "rating",
            "rating_text",
            "rating_chips",
            "token_use",
            "chat_id",
            "user_id",
        )


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = (
            "id",
            "email",
            "ai_experience",
            "business_unit",
            "grade",
            "uk_or_us_english",
            "profession",
            "role",
            "is_staff",
            "is_active",
            "is_superuser",
            "last_login",
        )
