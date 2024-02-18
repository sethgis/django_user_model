
from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path, include
from . import views
from users import views as user_views



urlpatterns = [
    path('', views.home, name='blog-home'),
    path('register/', user_views.register, name='register'),
    path('profile/', user_views.profile, name='profile'),
    # path('profile/', auth_views.LoginView.as_view(template_name='users/profile.html'), name='profile'),
    path('logout/', auth_views.LoginView.as_view(template_name='users/logged_out.html'), name='logout'),
    path('login/', auth_views.LoginView.as_view(template_name='users/login.html'), name='login'),
    path('seth/', views.home2, name='blog-home2'),
    path('about/', views.about, name='blog-about'),

]