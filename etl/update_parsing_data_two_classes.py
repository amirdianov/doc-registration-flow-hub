import datetime
import os
import shutil
import time
from pathlib import Path

from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor

from config import input_parsing_update, output_parsing_update, monitoring_path
from etl.utils.test_png_files import check_parsed_data


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
            with ProcessPoolExecutor(max_workers=4) as executor:
                for page in range(1, len(pages) - 1):
                    intermediate_page_path = os.path.join(output_folder, '0', f'{page + 1} - {pdf_file}.png')
                    executor.submit(save_page, pages[page], intermediate_page_path)


def convert_pdfs_in_folder(input_folder, output_folder):
    """Конвертировать все PDF в папке."""
    pdf_files = os.listdir(input_folder)

    with ProcessPoolExecutor(max_workers=6) as executor:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_folder, pdf_file)
            executor.submit(pdf_to_images, pdf_path, pdf_file, output_folder)


def move_files(src_dir, dst_dir):
    for file_name in os.listdir(src_dir):
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)

        # Проверяем, является ли элемент файлом (а не папкой)
        if os.path.isfile(src_file):
            shutil.move(src_file, dst_file)
        else:
            print(f"Skiped file")


if __name__ == '__main__':
    # Распределение файлов по соответствующим папкам, внутри каждой папки классы (0, 1)
    start_time = time.time()

    current_date = datetime.datetime.now().strftime("%d_%m_%Y")
    input_path = Path(input_parsing_update, f'update_{current_date}')
    output_path = Path(output_parsing_update, f'update_{current_date}')

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    move_files(monitoring_path, input_path)

    convert_pdfs_in_folder(input_path, output_path)
    check_parsed_data(
        Path(output_parsing_update),
        num_threads=10,
        delete_bad_files=True  # Удалять ли проблемные файлы
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"End: {elapsed_time:.2f} sec.")
    print('True')
