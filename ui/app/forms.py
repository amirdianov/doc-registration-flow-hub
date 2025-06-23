from django import forms

from app.models import GetFileText, SplitFile, NerToXML, GPTModel, FullProcessModel


class BaseForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_suffix = ""
        for attr, value in self.fields.items():
            input_class = "form-select" if self.fields[attr].widget.input_type == "select" else "form-control"
            self.fields[attr].widget.attrs.update({"class": input_class, "placeholder": "smt"})
            self.fields[attr].widget.attrs.update({'required': True})


class FullProcessForm(BaseForm):
    class Meta:
        model = FullProcessModel
        fields = '__all__'


class SplitFileForm(BaseForm):
    class Meta:
        model = SplitFile
        fields = '__all__'
        widgets = {
            'file': forms.ClearableFileInput(attrs={
                'accept': '.pdf',  # Браузер будет фильтровать файлы
                'class': 'form-control'
            })
        }


class GetFileTextForm(BaseForm):
    class Meta:
        model = GetFileText
        fields = '__all__'
        widgets = {
            'file': forms.ClearableFileInput(attrs={
                'accept': '.pdf',  # Браузер будет фильтровать файлы
                'class': 'form-control'
            })
        }


class NERToXMLForm(BaseForm):
    class Meta:
        model = NerToXML
        fields = '__all__'
        widgets = {
            'file': forms.ClearableFileInput(attrs={
                'accept': '.txt',  # Браузер будет фильтровать файлы
                'class': 'form-control'
            })
        }


class GPTModelForm(BaseForm):
    class Meta:
        model = GPTModel
        fields = '__all__'
        widgets = {
            'extra_parameters': forms.TextInput(attrs={
                'class': 'form-control',  # Стилизация Bootstrap
            }),
            'file': forms.ClearableFileInput(attrs={
                'accept': '.txt',  # Браузер будет фильтровать файлы
                'class': 'form-control'
            })
        }
