from django.db import models
from django.contrib.auth.models import User

class Profile(models.Models):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    
