from django.db import models

class Dataset(models.Model):
    name = models.CharField(max_length=100)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class AlgorithmResult(models.Model):
    algorithm_name = models.CharField(max_length=50)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.algorithm_name} - {self.dataset.name}"
