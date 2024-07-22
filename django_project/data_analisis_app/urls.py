from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home')
    # path('dashboard2/', views.dashboard2, name='dashboard2'),
    # path('dashboard3/', views.dashboard3, name='dashboard3'),
    # path('dashboard4/', views.dashboard4, name='dashboard4'),
]
