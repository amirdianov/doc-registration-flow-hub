import os

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Подключение к Minio ----------------------------------------------------------------
mlflow.set_tracking_uri("http://localhost:5000")  # или http://mlflow:5000 из контейнера
mlflow.set_experiment("s3_test")
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"  # если ты с хоста

def train_and_register_model():
    """Обучает модель и регистрирует её в MLflow."""
    # Генерация данных
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Обучение модели
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    signature = infer_signature(X_train, y_pred)

    # Логирование в MLflow
    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test[:5]
        )

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        client = MlflowClient()
        # client.create_registered_model("MyModel") # если первый запуск !!!
        new_version = client.create_model_version(name="MyModel", source=model_uri, run_id=run_id)
        
        # Обновление тегов версий
        for mv in client.search_model_versions(f"name='MyModel'"):
            if mv.tags.get("stage") == "production":
                prod_version = mv
                break
        client.delete_model_version_tag(
            name="MyModel",
            version=prod_version.version,
            key="stage"
        )
        client.set_model_version_tag(
            name="MyModel",
            version=new_version.version,
            key="stage",
            value="production"
        )
        print("Model registered in S3!")

def load_production_model():
    """Загружает модель с тегом production."""
    client = MlflowClient()
    model_name = "MyModel"
    tag_key = "stage"
    tag_value = "production"

    # Получаем все версии модели
    all_versions = client.search_model_versions(f"name='{model_name}'")

    for version in all_versions:
        # Получаем теги для версии модели
        tags = client.get_model_version(name="MyModel", version=version.version).tags
        print(tags)
        if 'production' in tags.values():
            model_uri = f"models:/{model_name}/{version.version}"
            loaded_model = mlflow.sklearn.load_model(model_uri)
            return loaded_model
    return None

def get_model_metrics(model_name="EfficientNet_B4_Document_Classifier"):
    """Получает метрики модели по тегу production."""
    client = MlflowClient()
    all_versions = client.search_model_versions(f"name='{model_name}'")

    model_version = None
    for version in all_versions:
        tags = client.get_model_version(name=model_name, version=version.version).tags
        print(tags)
        if 'production' in tags.values():
            model_version = version

    if model_version:
        run_id = model_version.run_id
        run_data = client.get_run(run_id).data
        test_acc = run_data.metrics.get("test_accuracy")
        print(f"Test accuracy for version {model_version}: {test_acc}")
        return test_acc
    return None

if __name__ == "__main__":
    # Обучаем и регистрируем модель
    train_and_register_model()
    
    # Загружаем production модель
    model = load_production_model()
    if model:
        print("Production model loaded successfully")
    
    # Получаем метрики
    get_model_metrics() 