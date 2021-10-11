from django.db import models

# Create your models here.
class mapImage(models.Model):
    file = models.FileField(blank=False, null=False, upload_to='media/')
    name = models.CharField(max_length = 255, default='img.jpg')
    uploaded_at = models.DateTimeField(auto_now_add=True)