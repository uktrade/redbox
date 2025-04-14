import logging
import uuid

import boto3
from django.conf import settings
from django.contrib.auth import get_user_model
from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAdminUser, IsAuthenticated

from redbox_app.redbox_core.models import ChatMessage
from redbox_app.redbox_core.serializers import ChatMessageSerializer, UserSerializer

User = get_user_model()

logger = logging.getLogger(__name__)


class StandardResultsSetPagination(PageNumberPagination):
    page_size = 50
    page_size_query_param = "page_size"
    max_page_size = 100


@api_view(["GET"])
@permission_classes([IsAuthenticated, IsAdminUser])
def user_view_pre_alpha(request):
    """Return paginated user data"""
    paginator = StandardResultsSetPagination()
    queryset = User.objects.all()
    result_page = paginator.paginate_queryset(queryset, request)
    serializer = UserSerializer(result_page, many=True, read_only=True)
    return paginator.get_paginated_response(serializer.data)


@api_view(["GET"])
@permission_classes([IsAuthenticated, IsAdminUser])
def message_view_pre_alpha(request):
    """Return paginated message data"""
    paginator = StandardResultsSetPagination()
    queryset = ChatMessage.objects.all()
    result_page = paginator.paginate_queryset(queryset, request)
    serializer = ChatMessageSerializer(result_page, many=True, read_only=True)
    return paginator.get_paginated_response(serializer.data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def aws_credentials_api():
    """Get credentials for AWS (used for transcription so far)"""

    client = boto3.client("sts")
    role_arn = settings.AWS_TRANSCRIBE_ROLE_ARN

    # Creating new credentials unfortunately sometimes fails
    max_attempts = 3
    for i in range(3):
        try:
            credentials = client.assume_role(
                RoleArn=role_arn,
                RoleSessionName="redbox_" + str(uuid.uuid4()),
                DurationSeconds=60 * 15,  # 15 minutes
            )["Credentials"]
        except Exception:
            if i == max_attempts - 1:
                raise

    return JsonResponse(
        {
            "AccessKeyId": credentials["AccessKeyId"],
            "SecretAccessKey": credentials["SecretAccessKey"],
            "SessionToken": credentials["SessionToken"],
            "Expiration": credentials["Expiration"],
        },
        status=200,
    )
