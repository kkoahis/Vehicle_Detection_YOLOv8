from django.db import models

class Vehicle(models.Model):
    license = models.CharField(max_length=20, unique=True)
    color = models.CharField(max_length=50)
    type = models.CharField(max_length=50)
    image = models.BinaryField()
    datetime = models.DateTimeField()
    location = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.license} - {self.color} {self.type}"
