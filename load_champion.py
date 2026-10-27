# ===== 1. Importer les bibliothèques =====
import mlflow
import mlflow.sklearn
import pandas as pd

# ===== 2. Se connecter à la même base MLflow =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 3. Charger le modèle champion avec son alias =====
# On ne connaît ni la version ni le run_id
model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)

print("Modèle chargé :", type(model).__name__)

# Afficher les variables attendues par le modèle
print(
    "Variables attendues :",
    list(model.feature_names_in_)
)

# ===== 4. Créer un nouveau client =====
# Les colonnes d'abonnement sont déjà transformées en 0/1
# Ici, le client possède un abonnement Standard
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])

# ===== 5. Garder uniquement les colonnes connues du modèle =====
# Et respecter exactement leur ordre
client = client[list(model.feature_names_in_)]

# ===== 6. Faire la prédiction =====
# 1 = le client va partir
# 0 = le client reste

prediction = model.predict(client)[0]

print(
    "Le client va-t-il partir ? ->",
    "OUI" if prediction == 1 else "NON"
)