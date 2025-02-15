from django.urls import path

from predict_handwriting.views import predict,index

urlpatterns = [
    path("", index, name="index"),
    path("predict", predict, name="predict"),
]