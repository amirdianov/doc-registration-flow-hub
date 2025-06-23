import os

from django.conf.global_settings import MEDIA_ROOT
from django.db import models


# Create your models here.


class BaseModelTask(models.Model):
    file = models.FileField(
        null=True,
        blank=True,
        verbose_name="Файл",
    )


class FullProcessModel(BaseModelTask):
    pass


class SplitFile(BaseModelTask):
    pass


class GetFileText(BaseModelTask):
    TYPE_CHOICE = (('Scan', 'Отсканированный pdf'), ('PDF', 'Реальный pdf'))

    type_file = models.CharField(choices=TYPE_CHOICE, max_length=20, verbose_name='Тип файла')


class NerToXML(BaseModelTask):
    PREPROCESS_CHOICE = (('0', 'Нет'), ('1', 'Да'))

    # is_preprocess = models.CharField(choices=PREPROCESS_CHOICE, max_length=20, verbose_name='Улучшить качество')


class GPTModel(BaseModelTask):
    PROMPT_TYPE = (('ner', 'Получить сущности'), ('answer', 'Задать вопросы'), ('apply_decision', 'Принять решение'))

    prompt_type = models.CharField(choices=PROMPT_TYPE, max_length=20, verbose_name='Задача')
    extra_parameters = models.TextField(verbose_name='Дополнительные параметры')
