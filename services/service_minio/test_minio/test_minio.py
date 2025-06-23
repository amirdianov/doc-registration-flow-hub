import boto3
from botocore.client import Config

# Настройки MinIO (из docker-compose.yml)
MINIO_ENDPOINT = "localhost:9000"  # или IP сервера, если не локально
MINIO_ACCESS_KEY = "admin"  # ваш MINIO_ROOT_USER
MINIO_SECRET_KEY = "password"  # ваш MINIO_ROOT_PASSWORD
BUCKET_NAME = "test-bucket"  # создайте бакет через веб-интерфейс

# Инициализация клиента для MinIO
s3 = boto3.client(
    "s3",
    endpoint_url=f"http://{MINIO_ENDPOINT}",  # http (для HTTPS укажите https://)
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version="s3v4"),  # важно для MinIO
    region_name="us-east-1"  # MinIO требует регион (любой)
)


def upload_file(local_path, s3_key):
    """Загружает файл в MinIO."""
    try:
        s3.upload_file(local_path, BUCKET_NAME, s3_key)
        print(f"Файл '{local_path}' загружен в '{s3_key}'.")
    except Exception as e:
        print(f"Ошибка загрузки: {e}")


def download_file(s3_key, local_path):
    """Скачивает файл из MinIO."""
    try:
        s3.download_file(BUCKET_NAME, s3_key, local_path)
        print(f"Файл '{s3_key}' скачан в '{local_path}'.")
    except Exception as e:
        print(f"Ошибка скачивания: {e}")


def create_test_file(content="Test content"):
    """Создает тестовый файл."""
    with open("my_file.txt", "w") as f:
        f.write(content)
    print("Тестовый файл создан")


# Пример использования
if __name__ == "__main__":
    # Создаем тестовый файл
    create_test_file()
    
    # Пример загрузки
    upload_file("my_file.txt", "folder/in/service_minio/my_file.txt")

    # Пример скачивания
    download_file("folder/in/service_minio/my_file.txt", "downloaded_file.txt")