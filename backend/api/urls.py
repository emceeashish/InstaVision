from django.urls import path
from .views import ProfileView, PostsView, image_proxy

urlpatterns = [
    path('profile/<str:username>/', ProfileView.as_view(), name='api-profile'),
    path('posts/<str:username>/', PostsView.as_view(), name='api-posts'),
    path('image-proxy/', image_proxy, name='api-image-proxy'),
]
