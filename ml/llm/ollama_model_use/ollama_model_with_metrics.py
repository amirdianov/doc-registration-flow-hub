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


    def _read_file_with_encoding(self, file_path: str) -> str:
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

    def _log_mlflow_metrics(self, metrics: Dict[str, float], results: List[Dict], prefix: str = ""):
        """Логирует метрики в MLflow"""
        # Логируем основные метрики
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{prefix}{metric_name}", value)

    def _process_document_pair(self, text_file: str, decision_file: str, process_func) -> Dict:
        """Обрабатывает пару файлов (текст и решение)"""
        try:
            document_text = self._read_file_with_encoding(text_file)
            decision_text = self._read_file_with_encoding(decision_file)

            expected, predicted = process_func(document_text, decision_text)

            is_correct = self._compare_texts(predicted, expected)

            return {
                'file': text_file,
                'expected': expected,
                'predicted': predicted,
                'is_correct': is_correct
            }
        except Exception as e:
            print(f"Ошибка при обработке файлов {text_file} и {decision_file}: {str(e)}")
            return None

    def _compare_texts(self, predicted: str, expected: str) -> bool:
        """Сравнивает предсказанный текст с эталонным"""
        predicted = predicted.lower().strip()
        expected = expected.lower().strip()

        return predicted == expected

    def _extract_document_type(self, decision_text: str) -> str:
        """Извлекает тип документа из текста решения"""
        match = re.search(r'Тип документа:\s*(.*?)(?:\n|$)', decision_text)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_decision(self, decision_text: str) -> str:
        """Извлекает результат из текста решения"""
        match = re.search(r'Результат:\s*(.*?)(?:\n|$)', decision_text)
        if match:
            return match.group(1).strip()
        return ""

    def test_model_quality(self, test_dir: str) -> Dict[str, Dict]:
        """
        Тестирование качества модели (тип документа и результат)

        Args:
            test_dir: Путь к директории с тестовыми документами

        Returns:
            Dict с результатами тестирования
        """
        # Создаем основной эксперимент
        # Устанавливаем текущий эксперимент
        mlflow.set_tracking_uri("http://localhost:5000")  # или http://mlflow:5000 из контейнера
        mlflow.set_experiment("DocumentTypeClassification_Metrics")
        # Собираем все документы
        documents = []
        for root, dirs, files in os.walk(test_dir):
            text_file, decision_file = self._find_document_pair(files, root)
            if text_file and decision_file:
                try:
                    document_text = self._read_file_with_encoding(text_file)
                    decision_text = self._read_file_with_encoding(decision_file)
                    documents.append({
                        'text_file': text_file,
                        'document_text': document_text,
                        'decision_text': decision_text,
                        'expected_type': self._extract_document_type(decision_text),
                        'expected_decision': self._extract_decision(decision_text)
                    })
                except Exception as e:
                    print(f"Ошибка при чтении файлов {text_file} и {decision_file}: {str(e)}")
                    continue

        with mlflow.start_run(run_name="main_test_run") as main_run:
            # Логируем параметры модели
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("test_directory", test_dir)

            # Тестируем определение типа документа для всех документов
            type_results = []
            type_correct = 0

            with mlflow.start_run(run_name="document_type_test", nested=True) as type_run:
                for doc in documents:
                    predicted_type = self._determine_document_type(doc['document_text'])
                    is_correct = self._compare_texts(predicted_type, doc['expected_type'])
                    type_correct += int(is_correct)

                    result = {
                        'file': doc['text_file'],
                        'expected': doc['expected_type'],
                        'predicted': predicted_type,
                        'is_correct': is_correct
                    }
                    type_results.append(result)

                    # Сохраняем предсказанный тип для использования при определении результата
                    doc['predicted_type'] = predicted_type

                # Логируем метрики типа документа
                type_accuracy = type_correct / len(documents) if documents else 0
                mlflow.log_metric("accuracy", type_accuracy)
                mlflow.log_metric("total_documents", len(documents))
                mlflow.log_metric("correct_predictions", type_correct)


            # Тестируем определение результата для всех документов
            decision_results = []
            decision_correct = 0

            with mlflow.start_run(run_name="document_decision_test", nested=True) as decision_run:
                for doc in documents:
                    # Используем уже определенный тип документа
                    config = {
                        "role": "юридических и судебных делах",
                        "prompt_type": "apply_decision",
                        "document_text": doc['document_text'],
                        "document_type": doc['predicted_type']
                    }
                    predicted_decision = self.generate_text(config)
                    is_correct = self._compare_texts(self._extract_decision(predicted_decision), doc['expected_decision'])
                    decision_correct += int(is_correct)

                    result = {
                        'file': doc['text_file'],
                        'expected': doc['expected_decision'],
                        'predicted': predicted_decision,
                        'is_correct': is_correct
                    }
                    decision_results.append(result)

                # Логируем метрики результата
                decision_accuracy = decision_correct / len(documents) if documents else 0
                mlflow.log_metric("accuracy", decision_accuracy)
                mlflow.log_metric("total_documents", len(documents))
                mlflow.log_metric("correct_predictions", decision_correct)


            # Логируем итоговые метрики в основной run
            mlflow.log_metric("type_accuracy", type_accuracy)
            mlflow.log_metric("decision_accuracy", decision_accuracy)
            mlflow.log_metric("total_documents", len(documents))
            mlflow.log_metric("type_correct", type_correct)
            mlflow.log_metric("decision_correct", decision_correct)

            return {
                'document_type': {
                    'metrics': {
                        'accuracy': type_accuracy,
                        'total_documents': len(documents),
                        'correct_predictions': type_correct
                    },
                    'results': type_results
                },
                'decision': {
                    'metrics': {
                        'accuracy': decision_accuracy,
                        'total_documents': len(documents),
                        'correct_predictions': decision_correct
                    },
                    'results': decision_results
                }
            }

    def _find_document_pair(self, files: List[str], root: str) -> Tuple[str, str]:
        """Находит пару файлов (текст и решение)"""
        text_file = None
        decision_file = None

        for file in files:
            if file.endswith('.txt'):
                if 'desition' in file.lower():
                    decision_file = os.path.join(root, file)
                else:
                    text_file = os.path.join(root, file)

        return text_file, decision_file


if __name__ == '__main__':
    model = Model(base_url='http://localhost:11434/v1')
    # os.environ["AWS_ACCESS_KEY_ID"] = ""
    # os.environ["AWS_SECRET_ACCESS_KEY"] = ""
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    # Тестирование качества модели
    test_dir = r"\\192.168.0.89\c$\Users\a.diyanov\Documents\ДокументыДляАнализа"
    results = model.test_model_quality(test_dir)

    # Вывод результатов тестирования типа документа
    type_results = results['document_type']
    print("\nРезультаты тестирования определения типа документа:")
    print(f"Точность: {type_results['metrics']['accuracy']:.2%}")
    print(f"Всего документов: {type_results['metrics']['total_documents']}")
    print(f"Правильных предсказаний: {type_results['metrics']['correct_predictions']}")

    # Вывод результатов тестирования результата
    decision_results = results['decision']
    print("\nРезультаты тестирования определения результата:")
    print(f"Точность: {decision_results['metrics']['accuracy']:.2%}")
    print(f"Всего документов: {decision_results['metrics']['total_documents']}")
    print(f"Правильных предсказаний: {decision_results['metrics']['correct_predictions']}")

    # Вывод детальных результатов
    print("\nДетальные результаты:")
    for i, (type_result, decision_result) in enumerate(zip(type_results['results'], decision_results['results'])):
        print(f"\nДокумент {i + 1}:")
        print(f"Файл: {type_result['file']}")
        print(f"Тип документа:")
        print(f"  Ожидаемый: {type_result['expected']}")
        print(f"  Предсказанный: {type_result['predicted']}")
        print(f"  Правильно: {'Да' if type_result['is_correct'] else 'Нет'}")
        print(f"Результат:")
        print(f"  Ожидаемый: {decision_result['expected']}")
        print(f"  Предсказанный: {decision_result['predicted']}")
        print(f"  Правильно: {'Да' if decision_result['is_correct'] else 'Нет'}")
