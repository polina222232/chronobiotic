# chronobioticagent/main/api/v1/urls.py
"""
Main API v1 URL configuration
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter

router = DefaultRouter()

urlpatterns = [
    # Agent endpoints
    path('agents/', include('main.api.v1.agents.urls')),
    
    # Chat endpoints
    path('chat/', include('main.api.v1.chat.urls')),
    
    # Chemical endpoints
    path('chemical/', include('main.api.v1.chemical.urls')),
    
    # KAG endpoints
    path('kag/', include('main.api.v1.kag.urls')),
    
    # RAG endpoints
    path('rag/', include('main.api.v1.rag.urls')),
]

urlpatterns += router.urls
