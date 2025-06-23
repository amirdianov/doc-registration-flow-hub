import os
from abc import abstractmethod, ABC

from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from ml.llm.prompts import get_ner_prompt, get_summary_prompt, \
    get_answer_prompt

load_dotenv()


class BaseModel(ABC):
    @abstractmethod
    def __init__(self, model_path: str) -> None:
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
    def __init__(self, model_name: str = "mistralai/Mistral-Nemo-Instruct-2407") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        login(os.environ.get("HUGGING_FACE_KEY", ''))
        # Настройка квантования
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        # Загрузка модели и токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Установим pad_token_id, если его нет
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Используем eos_token как pad_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config
        )

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
    model_name: str = "mistralai/Mistral-Nemo-Instruct-2407"
    model = Model(model_name)

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
