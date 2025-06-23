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


def main(data_dir, num_threads=4):
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

    # Вывод результатов
    for file_info in results:
        if not file_info[1]:  # Если проверка не пройдена
            print(f"Проблемное изображение: {file_info[0]} — Ошибка: {file_info[2]}")

    print(f"Проверено {len(all_files)} файлов. Проблемных: {len([r for r in results if not r[1]])}")


if __name__ == "__main__":
    main(r'\\192.168.0.89\c$\Users\a.diyanov\PyCharmProjects\test\data_multi',
         num_threads=10)  # Укажите путь к вашим данным и число потоков
