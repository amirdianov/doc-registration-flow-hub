from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def check_image(file_path):
    """Функция для проверки целостности изображения."""
    try:
        with Image.open(file_path) as img:
            img.verify()  # Проверяет целостность файла
        return file_path, True
    except Exception as e:
        return file_path, False, str(e)


def check_parsed_data(data_dir, num_threads=4, delete_bad_files=False):
    """Многопоточная проверка всех изображений в указанной директории."""
    all_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            all_files.append(os.path.join(root, file))

    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_file = {executor.submit(check_image, file): file for file in all_files}
        for future in as_completed(future_to_file):
            result = future.result()
            results.append(result)

    # Вывод результатов и, если требуется, удаление проблемных файлов
    bad_files = [r for r in results if not r[1]]
    for file_info in bad_files:
        print(f"Проблемное изображение: {file_info[0]} — Ошибка: {file_info[2]}")
        if delete_bad_files:
            try:
                os.remove(file_info[0])
                print(f"Удалено: {file_info[0]}")
            except Exception as e:
                print(f"Ошибка удаления {file_info[0]}: {e}")

    print(f"Проверено {len(all_files)} файлов. Проблемных: {len(bad_files)}")