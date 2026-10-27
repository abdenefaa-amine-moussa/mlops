# ===== 1. Importer les bibliothèques =====
import mlflow
from mlflow import MlflowClient

# ===== 2. Se connecter à la même base que le TP3 =====
# Le fichier mlflow.db contient tous les runs
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 3. Retrouver le meilleur run =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# Première ligne = meilleur run
best = runs.iloc[0]

# Récupérer son identifiant
run_id = best["run_id"]

print(
    "Meilleur run :",
    best["tags.mlflow.runName"],
    "| id :",
    run_id[:8],
    "..."
)

# ===== 4. Enregistrer le modèle dans le Registry =====
# runs:/<run_id>/model = modèle enregistré dans ce run
# churn-model = nom du modèle dans le Registry
result = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)

print(
    "Modèle enregistré : churn-model | version",
    result.version
)

# ===== 5. Définir l'alias "champion" =====
client = MlflowClient()

client.set_registered_model_alias(
    "churn-model",
    "champion",
    result.version
)

print(
    "Alias 'champion' -> version",
    result.version
)