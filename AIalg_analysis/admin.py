from django.contrib import admin
from .models import Dataset, AlgorithmResult

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('name', 'uploaded_at')

@admin.register(AlgorithmResult)
class AlgorithmResultAdmin(admin.ModelAdmin):
    list_display = ('algorithm_name', 'dataset', 'accuracy', 'created_at')
