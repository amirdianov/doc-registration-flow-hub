from django.urls import path

from app.views import ServiceView

urlpatterns = [
    path('', ServiceView.as_view(), name='service_view'),
]
