from django.contrib.auth import get_user_model
from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken

from redbox_app.redbox_core.serializers import UserSerializer

User = get_user_model()


@api_view(["GET"])
@permission_classes([IsAdminUser, IsAuthenticated])
def user_view_pre_alpha(request):
    """this is for testing and evaluation only
    this *will* change so that not all data is returned!
    """
    serializer = UserSerializer(User.objects.all(), many=True, read_only=True)
    return Response(serializer.data)


@api_view(["GET"])
@permission_classes([IsAdminUser, IsAuthenticated])
def issue_token_after_sso(request):
    user = request.user
    refresh = RefreshToken.for_user(user)
    return JsonResponse({
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    })