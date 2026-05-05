# chronobioticagent/main/api/urls.py
from django.urls import path

from main import views

urlpatterns = [
    path('chat/', views.chat_api, name='chat_api'),
    path('chat/stream/', views.chat_stream_api, name='chat_stream_api'),
    path('voice/transcribe/', views.voice_transcribe_api, name='voice_transcribe_api'),
    path('voice/synthesize/', views.voice_synthesize_api, name='voice_synthesize_api'),
    path('search/substance/', views.search_substance_api, name='search_substance_api'),
    path('citations/', views.get_citations_api, name='get_citations_api'),
]
