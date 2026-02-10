"""
ASGI config for chronobiotic project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

"""
ASGI config for chronobiotic project.

Enables Django + Channels (WebSockets).
"""

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path, include

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chronobiotic.settings")

django_app = get_asgi_application()

# Если будут WebSocket-консьюмеры: main.agent.consumers
try:
    from main.agent import consumers as agent_consumers
    websocket_urlpatterns = [
        path("ws/agent/", agent_consumers.AgentConsumer.as_asgi()),
    ]
except Exception:
    websocket_urlpatterns = []

application = ProtocolTypeRouter(
    {
        "http": django_app,
        "websocket": URLRouter(websocket_urlpatterns),
    }
)