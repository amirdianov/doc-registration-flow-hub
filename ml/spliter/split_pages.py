import os
from pathlib import Path

import img2pdf
from pdf2image import convert_from_path
from torchvision import transforms, models
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B4_Weights

from ml.spliter.utils.delete_blank_page import del_blank_pages


def save_pdf(images, output_dir, counter):
    # сохранение файла
    pdf_path = Path(f"{output_dir}", f"section_{counter}_from{images[0][0]}_to{images[-1][0]}.pdf")

    # Список для хранения путей к временным изображениям
    img_paths = []

    for index, page in enumerate(images):
        # Сохраняем изображение во временный файл
        page_in_merged_pdf, image = page
        img_path = Path(f"{output_dir}", f"page_{counter}_{index}.jpg")
        os.makedirs(output_dir, exist_ok=True)
        image.save(img_path, 'PNG', dpi=(200, 200))
        img_paths.append(img_path)

    # Конвертируем изображения в PDF
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(img_paths))

    # # Удаляем временные файлы
    for img_path in img_paths:
        os.remove(img_path)

    print(f"Saved PDF: {pdf_path}")


def predict_page_is_first(model, transform, device, image):
    """Функция возвращает True, если страница определена как первая."""
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor).squeeze()
        prediction = (torch.sigmoid(outputs) > 0.5).int()  # Преобразуем логиты в бинарные метки
    return prediction.item() == 1


def split_f(pdf_path: str, output_dir: str, loaded_model=None):
    # загрузка и подготовка модели
    if loaded_model:
        model = loaded_model
    else:
        model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 1)
        )

        model.load_state_dict(torch.load('best_model_efficientnet_b4_15.pth'))
    model.eval()  # Режим инференса
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Подготовка трансформаций, аналогичных обучению
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pages: iter = convert_from_path(pdf_path)
    current_section: list = []
    pdf_counter: int = 1

    for i, page in enumerate(pages):
        if i % 2 == 0:
            if predict_page_is_first(model, transform, device, page) and current_section:
                # Если текущий сегмент содержит страницы, сохраняем его в новый PDF
                current_section = del_blank_pages(current_section)
                save_pdf(current_section, output_dir, pdf_counter)
                pdf_counter += 1
                current_section = []  # Очистим для нового сегмента
            current_section.append((i + 1, page))
        else:
            current_section.append((i + 1, page))

    # Сохраняем последний сегмент, если он не пустой
    if current_section:
        current_section = del_blank_pages(current_section)
        save_pdf(current_section, output_dir, pdf_counter)


if __name__ == '__main__':
    pdf_path = r'\\192.168.0.89\c$\PackageForTest\potok14730.pdf'
    output_dir = 'tmp_spliter'  # Директория для сохранения секций PDF
    split_f(pdf_path, output_dir)
