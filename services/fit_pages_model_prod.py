import mlflow
import mlflow.pytorch
import os
import numpy as np
from mlflow import MlflowClient
from sklearn.metrics import f1_score, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
from torch.optim import Adam
import time  # Для замера времени
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from etl.config import output_parsing_train


def fit():
    # Трансформации для изображений
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Загружаем датасет
    data_dir = output_parsing_train
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Получаем метки классов
    targets = np.array(dataset.targets)

    # Получаем индексы для каждого класса
    class_0_indices = np.where(targets == 0)[0]
    class_1_indices = np.where(targets == 1)[0]

    # Перемешиваем индексы для каждого класса
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)

    # Выбираем по 3 примера для инференса из каждого класса
    inference_class_0_indices = np.random.choice(class_0_indices, 3, replace=False)
    inference_class_1_indices = np.random.choice(class_1_indices, 3, replace=False)

    # Удаляем выбранные индексы из обучающей выборки
    remaining_class_0_indices = np.setdiff1d(class_0_indices, inference_class_0_indices)
    remaining_class_1_indices = np.setdiff1d(class_1_indices, inference_class_1_indices)

    # Объединяем индексы и снова перемешиваем
    balanced_indices = np.hstack((remaining_class_0_indices, remaining_class_1_indices))
    np.random.shuffle(balanced_indices)

    # Стратифицированное разделение данных на тренировочные, валидационные и тестовые выборки
    train_indices, test_indices = train_test_split(balanced_indices, test_size=0.15, stratify=targets[balanced_indices])
    train_indices, val_indices = train_test_split(train_indices, test_size=0.15, stratify=targets[train_indices])

    # Создание подвыборок
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Даталоадеры
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Используем предобученную модель EfficientNet_B4
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

    # Изменение последнего слоя для бинарной классификации
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, 1)
    )

    # Устройство для обучения
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Критерий, оптимизатор и планировщик
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # Настройки обучения
    epochs = 30
    patience = 3
    best_val_loss = float('inf')
    patience_counter = 0
    total_training_time = 0

    # **Начало работы с MLflow**
    with mlflow.start_run():
        # Логирование гиперпараметров
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 0.0001)
        mlflow.log_param("weight_decay", 1e-5)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("scheduler", "ReduceLROnPlateau")
        mlflow.log_param("pretrained_model", "EfficientNet_B4")

        for epoch in range(epochs):
            epoch_start_time = time.time()
            model.train()
            running_loss = 0.0
            y_true, y_pred = [], []

            # Обучение
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                # Гарантируем одинаковую размерность, тк в батче м.б. один элемент и тогда вернется скаляр

                outputs = outputs.reshape(labels.shape)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                y_true.extend(labels.detach().cpu().numpy())
                y_pred.extend(torch.sigmoid(outputs).detach().cpu().numpy())

            train_f1 = f1_score(y_true, np.array(y_pred) > 0.5)

            # Валидация
            model.eval()
            val_loss = 0.0
            correct, total = 0, 0
            y_true, y_pred = [], []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.float().to(device)
                    outputs = model(inputs).squeeze()
                    val_loss += criterion(outputs, labels).item()

                    predicted = torch.sigmoid(outputs) > 0.5
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                    y_true.extend(labels.detach().cpu().numpy())
                    y_pred.extend(torch.sigmoid(outputs).detach().cpu().numpy())

            val_accuracy = correct / total * 100
            val_f1 = f1_score(y_true, np.array(y_pred) > 0.5)

            # Логирование метрик
            mlflow.log_metric("train_loss", running_loss / len(train_loader), step=epoch)
            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("val_loss", val_loss / len(val_loader), step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

            # Ранняя остановка
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Сохраните лучшую модель
                torch.save(model.state_dict(), 'best_model_efficientnet_b4_test_new_types.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

            scheduler.step(val_loss)
            total_training_time += time.time() - epoch_start_time
        torch.save(model.state_dict(), 'document_classifier_efficientnet_b4_15_types.pth')

        # Логирование времени обучения
        mlflow.log_metric("total_training_time", total_training_time)
        # Тестирование
        model.load_state_dict(torch.load('best_model_efficientnet_b4_test_new_types.pth'))
        model.eval()
        class_names = dataset.classes
        test_start = time.time()
        test_loss, test_acc, test_f1, report = test_model(model, test_loader, criterion, device, class_names)
        test_time = time.time() - test_start

        # Логирование тестовых метрик
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_time": test_time,
            "total_training_time": total_training_time
        })

        # Логирование classification report
        mlflow.log_text(report, "classification_report.txt")

        # Сохранение модели
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact('best_model_efficientnet_b4_test_new_types.pth')

        # # Регистрация модели
        # mlflow.pytorch.log_model(model, "model")
        prod_version = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model",
                                             "EfficientNet_B4_Document_Classifier")
        client = MlflowClient()
        client.set_model_version_tag(
            name="EfficientNet_B4_Document_Classifier",
            version=prod_version.version,
            key="stage",
            value="production"
        )
        print("Модель зарегистрирована в MLflow.")


def test_model(model, loader, criterion, device, class_names):
    loss = 0.0
    correct = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs).squeeze()
            loss += criterion(outputs, labels).item()

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(torch.sigmoid(outputs).detach().cpu().numpy())

    report = classification_report(
        y_true,
        np.array(y_pred) > 0.5,
        target_names=class_names,
        output_dict=False
    )

    return (
        loss / len(loader),
        correct / len(loader.dataset) * 100,
        f1_score(y_true, np.array(y_pred) > 0.5),
        report
    )


if __name__ == '__main__':
    mlflow.set_tracking_uri("http://localhost:5000")  # или http://mlflow:5000 из контейнера
    mlflow.set_experiment("DocumentClassification")
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "password"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    fit()
