from abc import abstractmethod, ABC
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from ml.llm.prompts import get_ner_prompt, get_summary_prompt, \
    get_answer_prompt

load_dotenv()


class BaseModel(ABC):
    @abstractmethod
    def __init__(self, path_to_model: str) -> None:
        """Инициализация модели и токенизатора"""
        pass

    @abstractmethod
    def _get_prompt(self, config: dict) -> str:
        """Генерация промпта на основе конфигурации"""
        pass

    @abstractmethod
    def generate_text(self, config: dict) -> str:
        """Генерация текста на основе конфигурации"""
        pass


class Model:
    def __init__(self, path_to_model: Path):
        # Выбор устройства
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(path_to_model)
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    def __get_prompt(self, config):
        prompt_type = config["prompt_type"]
        if prompt_type == 'ner':
            return get_ner_prompt(config)
        elif prompt_type == 'summary':
            return get_summary_prompt(config)
        elif prompt_type == 'answer':
            return get_answer_prompt(config)

    # Функция генерации текста
    def generate_text(self, config):
        prompt = self.__get_prompt(config)

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)  # Учитываем паддинг
        inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Переносим на GPU

        # Генерация с настройкой параметров
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            max_new_tokens=1000,
            repetition_penalty=1.2  # Штраф за повторение
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text


if __name__ == '__main__':
    path_to_model: Path = Path("./Mistral-Nemo-Instruct-2407_local")
    model = Model(path_to_model)

    config = {
        "role": "юридических и судебных делах",
        "document_type": "Постановление о возбуждении исполнительного производства",
        "prompt_type": "summary",
        "document_text": """""",  # текст документа
        "parameters": [],  # какие данные нужно извлечь из текста
        # какие данные нужно извлечь из текста
    }

    generated_text = model.generate_text(config)
    print("Generated Text:", generated_text)
