from django.conf.urls import url
from apps.views import PigPostureView


urlpatterns = [
    url('', PigPostureView.as_view(), name="pig_posture")
]