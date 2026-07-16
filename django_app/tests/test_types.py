from django.contrib.auth import get_user_model
from django.test import Client

from redbox_app.redbox_core.types import StreamingTextBuffer

User = get_user_model()


def test_streaming_text_buffer(client: Client, alice: User):
    # Given
    client.force_login(alice)
    messages = [
        "Good ",
        "after",
        "noon,",
        " Mr. Amor.",
    ]

    # When
    stream_buffer = StreamingTextBuffer()
    response = []

    for message in messages:
        response.append(stream_buffer.process(message))

    response.append(stream_buffer.flush())

    # Then
    assert "".join(response) == "".join(messages)
    assert response[0] == "Good "
    assert response[1] == ""
    assert response[2] == ""
    assert response[3] == "afternoon, Mr. "
    assert response[4] == "Amor."
