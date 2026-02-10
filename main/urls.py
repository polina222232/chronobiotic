from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('about', views.about, name='about'),
    path('substance/<slug:linkname>/', views.substance_detail, name='substance_detail'),
    path('get_synonyms/<str:linkname>/', views.get_synonyms, name='get_synonyms'),
    path('agent-chat/', views.agent_chat, name='agent_chat'),
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/chat/stream/', views.chat_stream, name='chat_stream'),
    path('api/search/', views.search_database, name='search_database'),
]