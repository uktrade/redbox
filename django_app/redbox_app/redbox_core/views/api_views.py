from django.contrib.auth import get_user_model
from redbox_app.redbox_core.models import ChatMessage
from redbox_app.redbox_core.serializers import (ChatMessageSerializer,
                                                UserSerializer)
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.response import Response

User = get_user_model()

class StandardResultsSetPagination(PageNumberPagination):
    page_size = 50
    page_size_query_param = 'page_size'
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
    """Return paginated user data"""
    paginator = StandardResultsSetPagination()
    queryset = ChatMessage.objects.all()
    result_page = paginator.paginate_queryset(queryset, request)
    serializer = ChatMessageSerializer(result_page, many=False, read_only=True)
    return paginator.get_paginated_response(serializer.data)
