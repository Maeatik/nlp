from django.contrib import admin
from django.urls import path

from nlp import views

urlpatterns = [
   # path("", views.index),
   # path('admin/', admin.site.urls),
    path('', views.calculate, name='calculate'),
    #path("postuser/", views.postuser),
]

