import mlflow
import mlflow.pytorch
import os
import numpy as np
from mlflow import MlflowClient
from sklearn.metrics import f1_score, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch
import torch.nn as nn
from torch.optim import Adam
import time
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from etl.config import output_parsing_train, output_parsing_update

def get_production_model_version():
    """Получение текущей production версии модели"""
    client = MlflowClient()
    model_name = "EfficientNet_B4_Document_Classifier"
    
    all_versions = client.search_model_versions(f"name='{model_name}'")
    for version in all_versions:
        tags = client.get_model_version(name=model_name, version=version.version).tags
        if 'production' in tags.values():
            return version
    return None

def get_model_accuracy(version):
    """Получение accuracy модели по версии"""
    client = MlflowClient()
    run_id = version.run_id
    run_data = client.get_run(run_id).data
    return run_data.metrics.get("test_accuracy")

def load_production_model():
    """Загрузка production версии модели"""
    client = MlflowClient()
    model_name = "EfficientNet_B4_Document_Classifier"
    
    prod_version = get_production_model_version()
    if not prod_version:
        raise ValueError("Не найдена production версия модели")
    
    model_uri = f"models:/{model_name}/{prod_version.version}"
    return mlflow.pytorch.load_model(model_uri), prod_version

def prepare_datasets():
    """Подготовка и объединение старых и новых данных"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Загружаем старые данные
    old_dataset = datasets.ImageFolder(root=output_parsing_train, transform=transform)
    
    # Загружаем новые данные
    new_dataset = datasets.ImageFolder(root=output_parsing_update, transform=transform)
    
    # Объединяем датасеты
    combined_dataset = ConcatDataset([old_dataset, new_dataset])
    
    # Разделяем на train/val/test
    total_size = len(combined_dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)
    
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = Subset(combined_dataset, train_indices)
    val_dataset = Subset(combined_dataset, val_indices)
    test_dataset = Subset(combined_dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset

def fine_tune():
    # Загружаем production модель и её версию
    model, current_prod_version = load_production_model()
    
    # Получаем текущую accuracy
    current_accuracy = get_model_accuracy(current_prod_version)
    print(f"Текущая accuracy production модели: {current_accuracy}")
    
    # Замораживаем все слои кроме последних
    for param in model.parameters():
        param.requires_grad = False
    
    # Размораживаем только последние слои для fine-tuning
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Подготавливаем данные
    train_dataset, val_dataset, test_dataset = prepare_datasets()
    
    # Создаем даталоадеры
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Настройка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Настройка оптимизатора только для размороженных параметров
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=0.000001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    criterion = nn.BCEWithLogitsLoss()
    
    # Настройки обучения
    epochs = 15
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    total_training_time = 0
    
    # Начало работы с MLflow
    with mlflow.start_run():
        # Логирование гиперпараметров
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 0.000001)
        mlflow.log_param("weight_decay", 1e-5)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("scheduler", "ReduceLROnPlateau")
        mlflow.log_param("training_strategy", "fine-tuning")
        mlflow.log_param("frozen_layers", "all_except_classifier")
        
        # Логируем количество обучаемых параметров
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_count = sum(p.numel() for p in model.parameters())
        mlflow.log_param("trainable_params_ratio", trainable_params_count / total_params_count)
        
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
                torch.save(model.state_dict(), 'best_model_efficientnet_b4_finetuned.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
            
            scheduler.step(val_loss)
            total_training_time += time.time() - epoch_start_time
            
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}, "
                  f"Train F1: {train_f1:.4f}, Val Loss: {val_loss / len(val_loader):.4f}, "
                  f"Val Accuracy: {val_accuracy:.2f}%, Val F1: {val_f1:.4f}")
        
        # Тестирование
        model.load_state_dict(torch.load('best_model_efficientnet_b4_finetuned.pth'))
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                outputs = model(inputs).squeeze()
                test_loss += criterion(outputs, labels).item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                y_true.extend(labels.detach().cpu().numpy())
                y_pred.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        
        test_accuracy = correct / total * 100
        test_f1 = f1_score(y_true, np.array(y_pred) > 0.5)
        
        # Логирование финальных метрик
        mlflow.log_metrics({
            "test_loss": test_loss / len(test_loader),
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "total_training_time": total_training_time
        })
        
        # Сохранение модели
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact('best_model_efficientnet_b4_finetuned.pth')
        
        # Регистрация новой версии модели
        new_version = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            "EfficientNet_B4_Document_Classifier"
        )
        
        # Проверяем, лучше ли новая версия
        if test_accuracy > current_accuracy:
            print(f"Новая версия лучше! Accuracy: {test_accuracy:.2f}% vs {current_accuracy:.2f}%")
            
            # Удаляем тег production со старой версии
            client = MlflowClient()
            client.delete_model_version_tag(
                name="EfficientNet_B4_Document_Classifier",
                version=current_prod_version.version,
                key="stage"
            )
            
            # Устанавливаем тег production на новую версию
            client.set_model_version_tag(
                name="EfficientNet_B4_Document_Classifier",
                version=new_version.version,
                key="stage",
                value="production"
            )
            print("Модель успешно обновлена в MLflow.")
        else:
            print(f"Новая версия хуже. Accuracy: {test_accuracy:.2f}% vs {current_accuracy:.2f}%")
            print("Оставляем текущую production версию.")

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("DocumentClassification")
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "password"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    fine_tune() 