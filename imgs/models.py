from django.db import models

EMB_CHOICES =[
    ('lsb','LSB'),
    ('dwt','DWT'),
    ('emd','EMD'),
    ('pvd','PVD'),
]
# Create your models here.
class Image(models.Model):
    message = models.CharField(max_length=50)
    Img = models.ImageField(upload_to='images/')
    choices = models.CharField(max_length=20,choices=EMB_CHOICES,default='0')
