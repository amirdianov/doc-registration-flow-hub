import os
from abc import ABC, abstractmethod
from pathlib import Path
import re
import mlflow
from typing import Tuple, List, Dict

import openai
from dotenv import load_dotenv

from ml.llm.prompts import get_ner_prompt, get_summary_prompt, get_answer_prompt, get_document_type_prompt, \
    get_document_decision_prompt

load_dotenv()  # Загрузка переменных окружения


class BaseModel(ABC):
    @abstractmethod
    def __init__(self, base_url: str, model_name: str) -> None:
        """Инициализация клиента OpenAI и конфигурации"""
        pass

    @abstractmethod
    def _get_prompt(self, config: dict) -> str:
        """Генерация промпта на основе конфигурации"""
        pass

    @abstractmethod
    def generate_text(self, config: dict) -> str:
        """Генерация текста на основе конфигурации"""
        pass


class Model(BaseModel):
    def __init__(self, base_url: str = "http://ollama:11434/v1", model_name: str = "gemma3:12b-it-qat") -> None:
        self.model_name = model_name
        self.client = openai.OpenAI(base_url=base_url, api_key="ollama")

    def _get_prompt(self, config):
        prompt_type = config.get("prompt_type")
        if prompt_type == "document_type":
            return get_document_type_prompt(config)
        elif prompt_type == "ner":
            return get_ner_prompt(config)
        elif prompt_type == "summary":
            return get_summary_prompt(config)
        elif prompt_type == "answer":
            return get_answer_prompt(config)
        elif prompt_type == "apply_decision":
            return get_document_decision_prompt(config)
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

    def _determine_document_type(self, document_text: str) -> str:
        """Определение типа документа"""
        config = {
            "role": "юридических и судебных делах",
            "document_text": document_text,
            "prompt_type": "document_type"
        }

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Ты помощник, хорошо разбирающийся в юридических документах."},
                {"role": "user", "content": self._get_prompt(config)}
            ],
            temperature=0.1,  # Низкая температура для более точного определения типа
            max_tokens=50
        )
        return response.choices[0].message.content.strip()

    def generate_text(self, config):
        # Всегда определяем тип документа, если он не указан
        if "document_type" not in config or not config["document_type"]:
            document_type = self._determine_document_type(config["document_text"])
            config["document_type"] = document_type

        # Выбираем промпт в зависимости от prompt_type
        prompt = self._get_prompt(config)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Ты помощник, хорошо разбирающийся в юридических документах."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            top_p=0.9,
            max_tokens=1000,
            frequency_penalty=0.2,
            presence_penalty=0.0
        )
        generated_text = response.choices[0].message.content.strip()
        return generated_text

    def test_document_type_classification(self, test_dir: str) -> Dict[str, float]:
        """
        Тестирование точности определения типа документа
        
        Args:
            test_dir: Путь к директории с тестовыми документами
            
        Returns:
            Dict с метриками точности
        """
        correct_predictions = 0
        total_documents = 0
        results = []

        # Устанавливаем текущий эксперимент
        mlflow.set_tracking_uri("http://localhost:5000")  # или http://mlflow:5000 из контейнера
        mlflow.set_experiment("DocumentTypeClassification")

        # Проходим по всем подпапкам
        for root, dirs, files in os.walk(test_dir):
            # Ищем текстовый файл и файл с решением
            text_file = None
            decision_file = None

            for file in files:
                if file.endswith('.txt'):
                    if 'desition' in file.lower():
                        decision_file = os.path.join(root, file)
                    else:
                        text_file = os.path.join(root, file)

            if text_file and decision_file:
                # Читаем текст документа
                with open(text_file, 'r', encoding='utf-8') as f:
                    document_text = f.read()

                # Читаем эталонный тип документа
                with open(decision_file, 'r', encoding='windows-1251') as f:
                    decision_text = f.read()
                    expected_type = self._extract_document_type(decision_text)

                # Получаем предсказание модели
                predicted_type = self._determine_document_type(document_text)

                # Сравниваем и считаем метрики
                is_correct = self._compare_document_types(predicted_type, expected_type)
                correct_predictions += int(is_correct)
                total_documents += 1

                results.append({
                    'file': text_file,
                    'expected': expected_type,
                    'predicted': predicted_type,
                    'is_correct': is_correct
                })

        # Вычисляем метрики
        accuracy = correct_predictions / total_documents if total_documents > 0 else 0

        # Логируем метрики в MLflow
        with mlflow.start_run():
            # Логируем параметры модели
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("test_directory", test_dir)

            # Логируем основные метрики
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("total_documents", total_documents)
            mlflow.log_metric("correct_predictions", correct_predictions)

        return {
            'accuracy': accuracy,
            'total_documents': total_documents,
            'correct_predictions': correct_predictions,
            'detailed_results': results
        }

    def _extract_document_type(self, decision_text: str) -> str:
        """Извлекает тип документа из текста решения"""
        match = re.search(r'Тип документа:\s*(.*?)(?:\n|$)', decision_text)
        if match:
            return match.group(1).strip()
        return ""

    def _compare_document_types(self, predicted: str, expected: str) -> bool:
        """Сравнивает предсказанный тип документа с эталонным"""
        # Приводим к нижнему регистру и убираем лишние пробелы

        return predicted.lower().strip() == expected.lower().strip()

    def test_document_decision_quality(self, test_dir: str) -> Dict[str, float]:
        """
        Тестирование качества результатов модели

        Args:
            test_dir: Путь к директории с тестовыми документами

        Returns:
            Dict с метриками качества
        """
        correct_predictions = 0
        total_documents = 0
        results = []

        # Устанавливаем текущий эксперимент
        mlflow.set_tracking_uri("http://localhost:5000")  # или http://mlflow:5000 из контейнера
        mlflow.set_experiment("DocumentTypeClassification")

        def read_file_with_encoding(file_path: str) -> str:
            """Читает файл с учетом разных кодировок"""
            encodings = ['utf-8', 'cp1251', 'windows-1251', 'ascii']

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Ошибка при чтении файла {file_path} с кодировкой {encoding}: {str(e)}")
                    continue

            raise ValueError(f"Не удалось прочитать файл {file_path} ни с одной из кодировок: {encodings}")

        # Проходим по всем подпапкам
        for root, dirs, files in os.walk(test_dir):
            # Ищем текстовый файл и файл с решением
            text_file = None
            decision_file = None

            for file in files:
                if file.endswith('.txt'):
                    if 'desition' in file.lower():
                        decision_file = os.path.join(root, file)
                    else:
                        text_file = os.path.join(root, file)

            if text_file and decision_file:
                try:
                    # Читаем текст документа
                    document_text = read_file_with_encoding(text_file)

                    # Читаем эталонное решение
                    decision_text = read_file_with_encoding(decision_file)
                    expected_decision = self._extract_decision(decision_text)

                    # Получаем предсказание модели
                    config = {
                        "role": "юридических и судебных делах",
                        "prompt_type": "apply_decision",
                        "document_text": document_text
                    }
                    predicted_decision = self.generate_text(config)
                    for line in predicted_decision.split('\n'):
                        if line.strip().startswith('Результат:'):
                            predicted_decision = line.strip().replace('Результат:', '').strip()

                    # Сравниваем и считаем метрики
                    is_correct = self._compare_decisions(predicted_decision, expected_decision)
                    correct_predictions += int(is_correct)
                    total_documents += 1

                    results.append({
                        'file': text_file,
                        'expected': expected_decision,
                        'predicted': predicted_decision,
                        'is_correct': is_correct
                    })
                except Exception as e:
                    print(f"Ошибка при обработке файлов {text_file} и {decision_file}: {str(e)}")
                    continue

        # Вычисляем метрики
        accuracy = correct_predictions / total_documents if total_documents > 0 else 0

        # Логируем метрики в MLflow
        with mlflow.start_run():
            # Логируем параметры модели
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("test_directory", test_dir)

            # Логируем основные метрики
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("total_documents", total_documents)
            mlflow.log_metric("correct_predictions", correct_predictions)

        return {
            'accuracy': accuracy,
            'total_documents': total_documents,
            'correct_predictions': correct_predictions,
            'detailed_results': results
        }

    def _extract_decision(self, decision_text: str) -> str:
        """Извлекает решение из текста"""
        # Ищем блок с решением после "Тип документа:"
        match = re.search(r'Результат:\s*(.*?)(?:\n|$)', decision_text)
        if match:
            return match.group(1).strip()
        return ""

    def _compare_decisions(self, predicted: str, expected: str) -> bool:
        """Сравнивает предсказанное решение с эталонным"""
        # Приводим к нижнему регистру и убираем лишние пробелы
        predicted = predicted.lower().strip()
        expected = expected.lower().strip()

        return predicted.lower().strip() == expected.lower().strip()


if __name__ == '__main__':
    model = Model(base_url='http://localhost:11434/v1')

    config = {
        "role": "юридических и судебных делах",
        "prompt_type": "apply_decision",
        "document_text": """"""
    }

    # generated_text = model.generate_text(config)
    # print("Generated Text:", generated_text)
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "password"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    # Тестирование точности определения типа документа
    test_dir = r"C:\Users\a.diyanov\Documents\ДокументыДляАнализа"
    results = model.test_document_type_classification(test_dir)

    print(f"Точность определения типа документа: {results['accuracy']:.2%}")
    print(f"Всего документов: {results['total_documents']}")
    print(f"Правильных предсказаний: {results['correct_predictions']}")

    # Вывод детальных результатов
    print("\nДетальные результаты:")
    for result in results['detailed_results']:
        print(f"\nФайл: {result['file']}")
        print(f"Ожидаемый тип: {result['expected']}")
        print(f"Предсказанный тип: {result['predicted']}")
        print(f"Правильно: {'Да' if result['is_correct'] else 'Нет'}")

    # Тестирование качества результатов
    test_dir = r"C:\Users\a.diyanov\Documents\ДокументыДляАнализа"
    results = model.test_document_decision_quality(test_dir)

    print(f"Точность определения результата: {results['accuracy']:.2%}")
    print(f"Всего документов: {results['total_documents']}")
    print(f"Правильных предсказаний: {results['correct_predictions']}")

    # Вывод детальных результатов
    print("\nДетальные результаты:")
    for result in results['detailed_results']:
        print(f"\nФайл: {result['file']}")
        print(f"Ожидаемый результат: {result['expected']}")
        print(f"Предсказанный результат: {result['predicted']}")
        print(f"Правильно: {'Да' if result['is_correct'] else 'Нет'}")
