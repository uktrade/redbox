from django.contrib.auth import get_user_model
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.response import Response

from redbox_app.redbox_core.serializers import UserSerializer

User = get_user_model()

@api_view(["GET"])
@permission_classes([IsAuthenticated, IsAdminUser])
def user_view_pre_alpha(request):
    """this is for testing and evaluation only
    this *will* change so that not all data is returned!
    """
    serializer = UserSerializer(User.objects.all(), many=True, read_only=True)
    return Response(serializer.data)
