# Используем официальный Python-образ
FROM python:3.11-slim

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Устанавливаем рабочую директорию
WORKDIR /ui

# Устанавливаем зависимости системы
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi
RUN python -m spacy download ru_core_news_md

# Копируем всё содержимое проекта в контейнер
RUN mkdir "../ml"
COPY .. .
#COPY . .

# Собираем статические файлы
RUN python manage.py collectstatic --noinput

# Открываем порт приложения
EXPOSE 8000

# Устанавливаем команду по умолчанию для запуска приложения
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
