import os

from airflow import DAG
from airflow.sensors.python import PythonSensor
from datetime import datetime, timedelta
import paramiko

# PATHS
MONITORING_PATH = os.environ.get("MONITORING_PATH", '')  # новые файлы для обновления модели

SCRIPT_PARSING_UPDATE_DATASET_PATH = os.environ.get("SCRIPT_PARSING_UPDATE_DATASET_PATH", '')  # скрипт для парсинга pdf
SCRIPT_MODEL_UPDATE_DATASET_PATH = os.environ.get("SCRIPT_MODEL_UPDATE_DATASET_PATH", '')  # скрипт для дообучения

# SSH CONNECT
SSH_HOST_GPU = os.environ.get("SSH_HOST_GPU", '')
SSH_HOST_CPU = os.environ.get("SSH_HOST_CPU", '')
SSH_USERNAME = os.environ.get("SSH_USERNAME", '')
SSH_PASSWORD = os.environ.get("SSH_PASSWORD", '')

# базовые аргуенты для запуска задач, их можно изменять
args = {
    'owner': 'root',  # владелец задачи
    'start_date': datetime(2024, 1, 1),  # дата начала выполнения задачи
    'retries': 3,  # Количество попыток
    'retry_delay': timedelta(minutes=5),  # Задержка между попытками
}


def ssh_exec_command(command: str, ssh_host: str) -> tuple:
    """Выполнение команд на сервере по SSH"""

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(
        hostname=ssh_host,
        username=SSH_USERNAME,
        password=SSH_PASSWORD
    )
    return ssh.exec_command(command), ssh


def check_file_count(ssh_host: str) -> bool:
    """Проверка необходимого количества файлов для дообучения модели"""

    command_info, ssh = ssh_exec_command(
        f"PowerShell -Command \"(Get-ChildItem -Path '{MONITORING_PATH}' | Measure-Object).Count\"",
        ssh_host)
    stdin, stdout, stderr = command_info

    file_count = int(stdout.read().strip())
    ssh.close()
    return file_count >= 1000


def execute_script(path_to_script: str, ssh_host: str):
    """Запуск Python скрипта"""

    command_info, ssh = ssh_exec_command(f'python {path_to_script}', ssh_host)
    stdin, stdout, stderr = command_info
    stdout_output = stdout.read().decode('utf-8')
    stderr_output = stderr.read().decode('utf-8')

    ssh.close()

    # Проверяем результат
    if "True" in stdout_output:  # Скрипт должен завершаться выводом True
        return True
    else:
        raise Exception(f"Ошибка в скрипте. Логи:\n{stderr_output}")


with DAG(
        'update_model',
        default_args=args,
        schedule_interval="0 20 * * 5",  # Запуск в пятницу в 20:00
        start_date=datetime(2024, 1, 1),
        catchup=False,
) as dag:
    # Таск для ожидания появления 5 файлов (использует PythonSensor)
    check_file_count = PythonSensor(
        task_id='check_file_count',
        python_callable=check_file_count,
        op_kwargs={'ssh_host': SSH_HOST_GPU},
        poke_interval=10,  # Проверка каждые 10 секунд
        timeout=3600,  # Максимальное время ожидания 1 час
    )

    # Таск для выполнения Python скрипта по парсингу данных для дообучения (использует PythonSensor)
    parsing_update = PythonSensor(
        task_id='parsing_update',
        python_callable=execute_script,
        op_kwargs={'path_to_script': SCRIPT_PARSING_UPDATE_DATASET_PATH, 'ssh_host': SSH_HOST_CPU},
        poke_interval=10,  # Проверка каждые 10 секунд
        timeout=3600,  # Максимальное время ожидания (1 час)
        retries=3,  # Повторять задачу до 3 раз
        retry_delay=timedelta(minutes=1),  # Задержка между попытками
        mode='poke',  # Режим работы сенсора
    )

    # Таск для выполнения Python скрипта по дообучению модели (использует PythonSensor)
    model_update = PythonSensor(
        task_id='model_update',
        python_callable=execute_script,
        op_kwargs={'path_to_script': SCRIPT_MODEL_UPDATE_DATASET_PATH, 'ssh_host': SSH_HOST_GPU},
        poke_interval=10,  # Проверка каждые 10 секунд
        timeout=3600,  # Максимальное время ожидания (1 час)
        retries=3,  # Повторять задачу до 3 раз
        retry_delay=timedelta(minutes=1),  # Задержка между попытками
        mode='poke',  # Режим работы сенсора
    )

    # Зависимости задач
    check_file_count >> parsing_update >> model_update
