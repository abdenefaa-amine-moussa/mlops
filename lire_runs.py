import mlflow

# Rechercher tous les runs de l'expérience "churn-prediction"
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# Colonnes à afficher
colonnes = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1"
]

print(runs[colonnes].to_string(index=False))

# La première ligne est le meilleur run
meilleur = runs.iloc[0]

print()

# Afficher le nom et les 8 premiers caractères de l'ID
print(
    "Meilleur run :",
    meilleur["tags.mlflow.runName"],
    "| id :",
    meilleur["run_id"][:8],
    "..."
)