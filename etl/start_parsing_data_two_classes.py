import os
import time  # Для измерения времени
from pathlib import Path

from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor

from config import input_parsing_train, output_parsing_train
from utils.test_png_files import check_parsed_data


need_copy = [str(i) for i in range(1, 16)]
mapping = {
    '1': 3000,
    '2': 3000,
    '3': 300,
    '4': 600,
    '5': 1000,
    '6': 1000,
    '7': 3000,
    '8': 3000,
    '9': 3000,
    '10': 1000,
    '11': 3000,
    '12': 3000,
    '13': 3000,
    '14': 1500,
    '15': 1000,
}


def save_page(image, image_path):
    """Сохранить страницу изображения в заданном пути."""
    image.save(image_path, 'PNG')


def pdf_to_images(pdf_path, pdf_file, output_folder):
    """Конвертация PDF в изображения."""
    pages = convert_from_path(pdf_path, dpi=300)

    os.makedirs(os.path.join(output_folder, '0'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, '1'), exist_ok=True)

    image_path = os.path.join(output_folder, '1', f'0 - {pdf_file}.png')
    pages[0].save(image_path, 'PNG')

    if len(pages) > 1:
        image_path = os.path.join(output_folder, '0', f'{len(pages)} - {pdf_file}.png')
        pages[-1].save(image_path, 'PNG')

        if len(pages) > 2:
            with ProcessPoolExecutor(max_workers=2) as executor:
                for page in range(1, len(pages) - 1):
                    intermediate_page_path = os.path.join(output_folder, '0', f'{page + 1} - {pdf_file}.png')
                    executor.submit(save_page, pages[page], intermediate_page_path)


def convert_pdfs_in_folder(input_folder, output_folder):
    """Конвертировать все PDF в папке."""
    pdf_files = os.listdir(input_folder)
    print(f"Обработка папки: {input_folder}")

    with ProcessPoolExecutor(max_workers=2) as executor:
        print(os.path.dirname(input_folder))
        print(os.path.basename(input_folder))
        last_folder = os.path.basename(input_folder).split('_')[0]
        for pdf_file in pdf_files[:mapping[last_folder]]:
            print(f"Обработка файла: {pdf_file}")
            pdf_path = os.path.join(input_folder, pdf_file)
            executor.submit(pdf_to_images, pdf_path, pdf_file, output_folder)


def parallel_convert_folders(input_folder, output_folder):
    """Обработка всех подпапок."""
    folders = os.listdir(input_folder)

    with ProcessPoolExecutor(max_workers=6) as executor:
        for folder in folders:
            if '14_pril_5000_Приложение к Письму от контрагента' not in folder:
                if folder.split('_')[0] in need_copy:
                    input_subfolder = os.path.join(input_folder, folder)
                    output_subfolder = os.path.join(output_folder)
                    executor.submit(convert_pdfs_in_folder, input_subfolder, output_subfolder)


if __name__ == '__main__':
    # Распределение файлов по соответствующим папкам, внутри каждой папки классы (0, 1)
    start_time = time.time()

    parallel_convert_folders(input_parsing_train, output_parsing_train)
    check_parsed_data(
        Path(output_parsing_train),
        num_threads=10,
        delete_bad_files=True  # Удалять ли проблемные файлы
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Программа завершена. Время выполнения: {elapsed_time:.2f} секунд.")
