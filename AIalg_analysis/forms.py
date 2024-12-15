from django import forms

class DatasetUploadForm(forms.Form):
    dataset = forms.FileField(label='Încărcați un fișier CSV')
