"""
URL configuration for chronobiotic project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from main import views

urlpatterns = [
    path('admin/', admin.site.urls),
                  path('', views.index, name='home'),
                  path('about/', views.about, name='about'),  # Добавьте слеш в конце
                  path('publications/', views.publications, name='publications'),
                  path('rawdata/', views.rawdata, name='rawdata'),
                  path('substance/<slug:linkname>/', views.substance_detail, name='substance_detail'),
                  path('get_synonyms/<str:linkname>/', views.get_synonyms, name='get_synonyms'),
                  path('agent-chat/', views.agent_chat, name='agent_chat'),
                  path('api/chat/', views.chat_api, name='chat_api'),
                  path('api/chat/stream/', views.chat_stream, name='chat_stream'),
                  path('api/search/', views.search_database, name='search_database'),
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
