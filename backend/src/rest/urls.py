from django.urls import path
from rest.views import HelloWorldView


urlpatterns = [path('hello/', HelloWorldView.as_view(), name='hello')]
