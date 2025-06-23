import os
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login

load_dotenv()


def save_model(model_name: str, saved_directory: Path):
    # Выбор устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Вход в Hugging Face (используй свой токен)
    login(os.environ.get("HUGGING_FACE_KEY", ''))

    # Настройка квантования
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Загрузка модели и токенизатора
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Установим pad_token_id, если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Используем eos_token как pad_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )

    # Сохраняем модель и токенизатор локально
    model.save_pretrained(saved_directory)
    tokenizer.save_pretrained(saved_directory)


if __name__ == "__main__":
    model_name: str = "mistralai/Mistral-Nemo-Instruct-2407"

    saved_directory: Path = Path("./Mistral-Nemo-Instruct-2407_local")
    saved_directory.mkdir(parents=True, exist_ok=True)

    save_model(model_name, saved_directory)
