# Splitter Module

Этот модуль отвечает за разделение многостраничных документов на отдельные страницы с использованием модели компьютерного зрения. В основе лежит использование предобученной модели EfficientNet-B4 для определения границ страниц и их последующего разделения.

## 📁 Структура проекта

```
.
├── utils/                          # Вспомогательные утилиты
│   └── ...
│
├── split_pages.py                  # Основной модуль разделения страниц
└── best_model_efficientnet_b4_15.pth  # Веса предобученной модели
```

## 🔁 Основной процесс

### 🧩 Компоненты

- `split_pages.py` — основной модуль, реализующий функционал разделения страниц с использованием модели EfficientNet-B4
- `best_model_efficientnet_b4_15.pth` — файл с весами предобученной модели
- `utils/` — директория с вспомогательными утилитами для обработки изображений

### 🗂️ Функциональность

| № | Компонент                        | Назначение                                                                                               |
|--:|----------------------------------|----------------------------------------------------------------------------------------------------------|
| 1 | split_pages.py                   | Реализация разделения страниц с использованием модели EfficientNet-B4                                    |
| 2 | best_model_efficientnet_b4_15.pth| Веса предобученной модели для определения границ страниц                                                 |
| 3 | utils/                           | Вспомогательные функции для обработки изображений и работы с моделью                                     |

### 🔄 Процесс обработки

1. **Предобработка изображения**:
   - Загрузка многостраничного документа
   - Нормализация изображения
   - Подготовка данных для модели

2. **Определение границ**:
   - Применение модели EfficientNet-B4
   - Определение границ страниц
   - Валидация результатов

3. **Разделение страниц**:
   - Разрезание изображения по границам
   - Сохранение отдельных страниц
   - Проверка качества разделения

## 🛠️ Вспомогательные утилиты

Модуль обеспечивает следующую функциональность:

- Работа с моделью EfficientNet-B4
- Обработка изображений
- Определение границ страниц
- Сохранение результатов

## 📝 Примечания

- Требуется установленный PyTorch
- Необходимы библиотеки для работы с изображениями (OpenCV, PIL)
- Рекомендуется использовать GPU для ускорения работы модели
- Доступно логирование всех этапов работы
- Поддерживается работа с различными форматами изображений
- Модель оптимизирована для работы с документами различного формата
- Возможна настройка параметров предобработки для улучшения качества разделения 