"""file_upload_sample URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
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
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from rest_framework.urlpatterns import format_suffix_patterns
from apps.views import AtobienView, AtobienView2, SeverityResults

urlpatterns = [
    url(r'atobien_api_result/(?P<pk>[0-9]+)', SeverityResults.as_view(), name='atobien_api_result'),
    path('atobien_disease_severity/', AtobienView.as_view(), name="atobien_disease_severity"),
    path('atobien_product_prediction/', AtobienView2.as_view(), name="atobien_product_prediction"),
    path('admin/', admin.site.urls),
]

urlpatterns = format_suffix_patterns(urlpatterns)