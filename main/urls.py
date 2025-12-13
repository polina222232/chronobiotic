
from django.urls import path
from . import views
urlpatterns = [

    path('',views.index, name='home'),
    path('about',views.about, name='about'),
    path('substance/<slug:linkname>/', views.substance_detail, name='substance_detail'),
    path('get_synonyms/<str:linkname>/', views.get_synonyms, name='get_synonyms'),
    # path('index/<int:chronobiotic_id>',views.chronobiotic),
]
