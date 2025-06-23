import datetime
import logging
import os
import shutil
import uuid
from pathlib import Path
import mlflow.sklearn
import mlflow
from django.conf import settings
from django.core.files import File
from django.utils.text import get_valid_filename
from mlflow.tracking import MlflowClient
import mlflow.pytorch

from django.http import FileResponse, JsonResponse
from django.shortcuts import render
from django.views import View
from django.core.files.storage import default_storage
from django.core.exceptions import ValidationError

from app.forms import SplitFileForm, GetFileTextForm, FullProcessForm, GPTModelForm, NERToXMLForm
from app.utils import send_file
from ml.llm.ollama_model_use.ollama_model import Model
from ml.ner.recog import spacy_entity, natasha_entity, make_xml
from ml.ocr.get_text import ExtractorFactory
from ml.spliter.split_pages import split_f
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def handle_file_response(func):
    """Декоратор для обработки ошибок при возврате FileResponse"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            return JsonResponse({
                'error': 'Internal Server Error',
                'message': 'Произошла ошибка при обработке файла'
            }, status=500)

    return wrapper


class ServiceView(View):
    def get(self, request):
        forms_data = [
            ("split_file", "splitFileModal", "Разделить файлы", SplitFileForm()),
            ("get_file_text", "getTextModal", "Распознать текст", GetFileTextForm()),
            ("get_result", "fullProcessModal", "Запуск бизнес-процесса", FullProcessForm()),
            ("get_xml_file", "nerXmlModal", "NER в XML", NERToXMLForm()),
            ("get_model_answer", "gptModelModal", "GPT модель", GPTModelForm()),
        ]

        context = {
            "forms_data": forms_data
        }
        return render(request, "app/main.html", context)

    def _save_file_from_request(self, operation_type: str, file: File):
        relative_dir = os.path.join(datetime.datetime.now().strftime("%Y-%m-%d"), operation_type,
                                    str(uuid.uuid4()))
        operation_result_directory = os.path.join(settings.MEDIA_ROOT, relative_dir)
        os.makedirs(operation_result_directory, exist_ok=True)

        # Фильтруем имя файла
        safe_filename = get_valid_filename(os.path.basename(file.name))
        relative_path = os.path.join(relative_dir, safe_filename)

        # Сохраняем файл, указывая относительный путь
        file_name = default_storage.save(relative_path, file)

        file = default_storage.open(file_name)
        file_url = default_storage.url(file_name)
        file_path = default_storage.path(file_name)

        return file, file_url, file_path, relative_dir, operation_result_directory

    def _get_model_to_split(self):
        mlflow.set_tracking_uri('http://localhost:5000' if settings.DEBUG == 'True' else "http://mlflow:5000")
        client = MlflowClient()
        model_name = "EfficientNet_B4_Document_Classifier"
        tag_value = "production"
        loaded_model = None

        all_versions = client.search_model_versions(f"name='{model_name}'")
        for version in all_versions:
            tags = client.get_model_version(name=model_name, version=version.version).tags
            if tag_value in tags.values():
                model_uri = f"models:/{model_name}/{version.version}"
                loaded_model = mlflow.pytorch.load_model(model_uri)

        return loaded_model

    def _ocr(self, request, file_path):
        extractor_factory = ExtractorFactory(request.POST.get('type_file', 'Scan'))
        extractor = extractor_factory.get_extractor()
        result_text = extractor.extract(file_path)

        return result_text

    def _ner(self, text):
        spacy_entity(text)
        natasha_entity(text)
        root = make_xml()

        return root

    def _llm(self, request, text):
        model = Model('http://localhost:11434/v1' if settings.DEBUG == 'True' else 'http://ollama:11434/v1')
        config = {
            "role": "юридических и судебных делах",  # в чем эксперт
            "document_type": "судебные документы",
            "prompt_type": request.POST.get('prompt_type', 'apply_decision'),
            "document_text": text,  # текст документа
            "parameters": request.POST.get('extra_parameters')
        }
        generated_text = model.generate_text(config)
        return generated_text

    @handle_file_response
    def post(self, request):
        try:
            operation = request.POST.get('operation')
            file = request.FILES['file']

            if operation == 'split_file':
                (file, file_url, file_path,
                 relative_dir, split_file_directory) = self._save_file_from_request('split_file', file)

                loaded_model = self._get_model_to_split()

                out_put_dir = os.path.join(split_file_directory, 'tmp_spliter')
                os.makedirs(out_put_dir, exist_ok=True)
                split_f(file_path, out_put_dir, loaded_model)

                shutil.make_archive(
                    os.path.join(split_file_directory, f'{file.name}_documents_zip'),
                    'zip',
                    out_put_dir)

                # Удаляем tmp папку, оставляем только результат
                shutil.rmtree(out_put_dir)

                return FileResponse(open(os.path.join(split_file_directory, f'{file.name}_documents_zip.zip'), 'rb'),
                                    as_attachment=True,
                                    filename=f'{file.name}_documents_zip.zip')

            elif operation == 'get_file_text':
                (file, file_url, file_path,
                 relative_dir, text_file_directory) = self._save_file_from_request('get_file_text', file)

                # OCR
                result_text = self._ocr(request, file_path)

                # Записываем результат в файл
                with open(os.path.join(text_file_directory, f'{file.name}_file_txt.txt'), 'w',
                          encoding='utf-8') as new_file:
                    new_file.write(result_text)

                return FileResponse(open(os.path.join(text_file_directory, f'{file.name}_file_txt.txt'), 'rb'),
                                    as_attachment=True,
                                    filename=f'{file.name}_file_txt.txt')

            elif operation == 'get_xml_file':
                (file, file_url, file_path,
                 relative_dir, xml_file_directory) = self._save_file_from_request('get_xml_file', file)

                # Получаем текст документа
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                # NER
                xml_structure = self._ner(text)

                # Записываем результат в файл
                xml_filename = Path(file_path).parent.joinpath(f'{file.name}_file_xml.xml')
                tree = ET.ElementTree(xml_structure)
                tree.write(xml_filename, encoding="utf-8", xml_declaration=True)

                return FileResponse(open(xml_filename, 'rb'),
                                    as_attachment=True,
                                    filename=f'{file.name}_file_xml.xml')

            elif operation == 'get_model_answer':
                (file, file_url, file_path,
                 relative_dir, answer_file_directory) = self._save_file_from_request('get_model_answer', file)

                # Получаем текст документа
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                # LLM
                generated_text = self._llm(request, text)

                # Записываем результат в файл
                with open(os.path.join(answer_file_directory, f'{file.name}_file_llm.txt'), 'w',
                          encoding='utf-8') as new_file:
                    new_file.write(generated_text.split('Ответ:')[1] if 'Ответ:' in generated_text else generated_text)

                return FileResponse(open(os.path.join(answer_file_directory, f'{file.name}_file_llm.txt'), 'rb'),
                                    as_attachment=True,
                                    filename=f'{file.name}_file_llm.txt')

            elif operation == 'get_result':
                (file, file_url, file_path,
                 relative_dir, result_dir) = self._save_file_from_request('full_pipeline', file)

                # # Шаг 1: split_file
                split_output_dir = os.path.join(result_dir, f'pages')
                os.makedirs(split_output_dir, exist_ok=True)

                loaded_model = self._get_model_to_split()
                split_f(file_path, split_output_dir, loaded_model)

                for img_file in os.listdir(split_output_dir):
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.pdf')):
                        continue

                    page_path = os.path.join(split_output_dir, img_file)
                    logger.info(f"[get_result] Processing: {img_file}")

                    file_info_path = os.path.join(split_output_dir, Path(page_path).stem)
                    os.makedirs(file_info_path, exist_ok=True)

                    # OCR
                    text = self._ocr(request, page_path)
                    text_path = os.path.join(file_info_path, f'text_file.txt')
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(text)

                    # # NER
                    xml_structure = self._ner(text)
                    xml_filename = os.path.join(file_info_path, f'xml_file.xml')
                    tree = ET.ElementTree(xml_structure)
                    tree.write(xml_filename, encoding="utf-8", xml_declaration=True)

                    # LLM
                    generated_text = self._llm(request, text)
                    llm_answer_path = os.path.join(file_info_path, f'llm_file.txt')
                    with open(llm_answer_path, 'w',
                              encoding='utf-8') as new_file:
                        new_file.write(
                            generated_text.split('Ответ:')[1] if 'Ответ:' in generated_text else generated_text)

                # Архивация результата
                zip_path = os.path.join(result_dir, f'{file.name}_result.zip')
                send_file(split_output_dir)
                shutil.make_archive(zip_path.replace('.zip', ''), 'zip', split_output_dir)

                # Удаляем внутри папки и инфрмацию к ним, оставляем только результат - архив
                # shutil.rmtree(split_output_dir)

                return FileResponse(open(zip_path, 'rb'),
                                    as_attachment=True,
                                    filename=os.path.basename(zip_path))
        except ValidationError as e:
            return JsonResponse({
                'error': 'Validation Error',
                'message': str(e)
            }, status=400)
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            return JsonResponse({
                'error': 'Internal Server Error',
                'message': 'Произошла ошибка при обработке файла'
            }, status=500)
