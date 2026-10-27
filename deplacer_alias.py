# ===== 1. Importer les bibliothèques =====
import mlflow
from mlflow import MlflowClient

# ===== 2. Se connecter à la base MLflow =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Créer un client MLflow
client = MlflowClient()

# ===== 3. Retrouver le run "essai-2" =====
# "essai-2" correspond au modèle le plus faible du TP3
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'essai-2'"
)

# Récupérer le run_id
run_id = runs.iloc[0]["run_id"]

print("Run trouvé :", run_id[:8], "...")

# ===== 4. Enregistrer le modèle =====
# Cela crée une nouvelle version de "churn-model"
v = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)

print("Nouvelle version créée :", v.version)

# ===== 5. Vérifier où pointe actuellement l'alias "champion" =====
actuel = client.get_model_version_by_alias(
    "churn-model",
    "champion"
)

print(
    "Champion actuel : version",
    actuel.version
)

# ===== 6. Déplacer l'alias "champion" vers la nouvelle version =====
client.set_registered_model_alias(
    "churn-model",
    "champion",
    v.version
)

print(
    "Après déplacement : champion -> version",
    client.get_model_version_by_alias(
        "churn-model",
        "champion"
    ).version
)

# ===== 7. Remettre l'alias "champion" sur l'ancienne bonne version =====
client.set_registered_model_alias(
    "churn-model",
    "champion",
    actuel.version
)

print(
    "Après retour : champion -> version",
    client.get_model_version_by_alias(
        "churn-model",
        "champion"
    ).version
)