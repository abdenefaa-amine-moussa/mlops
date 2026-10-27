# ===== 1. Importer la bibliothèque =====
import requests


# ===== 2. Adresse de notre API =====
URL = "http://localhost:8000"


# ===== 3. Vérifier que l'API est en vie =====
response = requests.get(URL + "/health")

print(
    "GET /health ->",
    response.json()
)


# ===== 4. Tester avec un premier client =====
# Client avec faible ancienneté et faible consommation
client = {
    "age": 35,
    "salaire": 2500,
    "abonnement": "Standard",
    "anciennete": 1,
    "consommation": 10
}

response = requests.post(
    URL + "/predict",
    json=client
)

print(
    "POST /predict (client 1) ->",
    response.status_code,
    response.json()
)


# ===== 5. Tester avec un deuxième client =====
# Client fidèle avec forte ancienneté et forte consommation
client2 = {
    "age": 50,
    "salaire": 4000,
    "abonnement": "Premium",
    "anciennete": 10,
    "consommation": 80
}

response = requests.post(
    URL + "/predict",
    json=client2
)

print(
    "POST /predict (client 2) ->",
    response.status_code,
    response.json()
)


# ===== 6. Tester une erreur volontaire =====
# L'âge doit être un entier, mais on envoie du texte
mauvais = dict(
    client,
    age="abc"
)

response = requests.post(
    URL + "/predict",
    json=mauvais
)

print(
    "POST /predict (age='abc') ->",
    response.status_code
)

# Afficher le message d'erreur retourné par FastAPI
print(
    "Message :",
    response.json()["detail"][0]["msg"]
)