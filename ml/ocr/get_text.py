import os
import shutil
from pathlib import Path

import PyPDF2
import cv2
import easyocr

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
from pdf2image import convert_from_path


class TextFromDocumentExtractor(ABC):
    @abstractmethod
    def extract(self, path_to_file: Path):
        pass


class PDFExtractor(TextFromDocumentExtractor):
    """Получение текста из реального PDF файла"""

    def extract(self, path_to_file: str):
        with open(path_to_file, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page in range(len(pdf_reader.pages)):
                page_obj = pdf_reader.pages[page]
                text += page_obj.extract_text()
            return text


class ScanExtractor(TextFromDocumentExtractor):
    """Получение текста из снимка/скана документа"""

    def rotate_image(self, path):
        image = cv2.imread(path)

        # Переводим изображение в градации серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Применяем бинаризацию
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Обнаруживаем края с помощью алгоритма Canny
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)

        # Находим линии с помощью преобразования Хафа
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        # Собираем углы наклона всех линий
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Вычисляем угол в градусах
            angles.append(angle)

        # Вычисляем медианный угол (чтобы исключить выбросы)
        median_angle = np.median(angles)
        print(f"Угол наклона текста: {median_angle:.2f} градусов")

        # Получаем размеры изображения
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)  # Центр изображения

        # Создаем матрицу поворота
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        # Применяем поворот
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Сохраняем результат
        cv2.imwrite(path, rotated)
        return path

    def binary(self, path):
        img = np.asarray(Image.open(path))
        h, w = img.shape[:2]
        new_h, new_w = int(h * 3), int(w * 3)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 3)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return img

    def remove_noise(self, img: np.ndarray):
        img = cv2.bitwise_not(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        return img

    def proc_im(self, path):
        path: str = self.rotate_image(path)
        img: np.ndarray = self.binary(path)
        img: np.ndarray = self.remove_noise(img)
        img = img.astype(np.uint8)  # Убедимся, что тип данных 8-битный
        if len(img.shape) == 2:  # Если изображение одноканальное
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Конвертируем в 3-канальное (BGR)
        cv2.imwrite(path, img)
        return path

    def save_scans(self, path_to_file: str):
        pages = convert_from_path(path_to_file, dpi=300)
        os.makedirs(Path(path_to_file).parent.joinpath('tmp_ocr'), exist_ok=True)
        for ind, page in enumerate(pages):
            page.save(Path(path_to_file).parent.joinpath('tmp_ocr', f'file_{ind}.png'), 'PNG')

    def extract(self, path_to_file: str):
        use_preprocessing = True
        text_detections = ''
        self.save_scans(path_to_file)

        for root, _, files in os.walk(Path(path_to_file).parent.joinpath('tmp_ocr')):
            for file in files:
                path_to_page = os.path.join(root, file)
                if use_preprocessing:
                    try:
                        path_to_page = self.proc_im(path_to_page)
                    except Exception as e:
                        print("Ошибка при предварительной обработке:", e)
                img = cv2.imread(path_to_page)
                reader = easyocr.Reader(['ru', 'en'], gpu=True)
                text_detections += ' '.join(reader.readtext(img, detail=0))

        # Удаляем временные файлы
        temporary = Path(path_to_file).parent.joinpath('tmp_ocr')
        shutil.rmtree(temporary)
        return text_detections


class ExtractorFactory:
    """Получени е экстрактора для извлечения текста"""
    _extractors = {
        "PDF": PDFExtractor(),
        "Scan": ScanExtractor()
    }

    def __init__(self, doc_type: str = 'Scan') -> None:
        self.doc_type = doc_type

    def get_extractor(self) -> TextFromDocumentExtractor:
        return self._extractors.get(self.doc_type)


if __name__ == '__main__':
    doc_type = 'Scan'
    content = Path('./test_scan.pdf')

    extractor_factory = ExtractorFactory(doc_type)
    extractor = extractor_factory.get_extractor()

    result = extractor.extract(content)
    print(result)
