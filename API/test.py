import requests
import pandas as pd

path: str = "/home/fowardelcac/mlApp/API/Dataset/Diabetes.csv"
url: str = "http://127.0.0.1:8000/"

files: dict = {"file": open(path, "rb")}  # Make sure to use the correct path

# Make the POST request
response = requests.post(url + "uploadfile/", files=files)
content = response.json()
df = pd.DataFrame(content["data"])
print(df.head())
print("Columns:", content["columns"])

# d = requests.get("http://127.0.0.1:8000/data/").json()["data"]
# print(pd.DataFrame(d).head())

body = {
    "columns": [
        "age",
        "sex",
        "bmi",
        "bp",
    ],
    "target": "Target",
}

split_post = requests.post(url + "split/", json=body)
content = split_post.json()
print(content)

loop_list = ["x_train", "x_test", "y_train", "y_test"]
for i in loop_list:
    print("Dataframe:", i)
    d = requests.get("http://127.0.0.1:8000/data/").json()[i]
    print(pd.DataFrame(d).head())
    print("*" * 100)

random_forest_params = {
    "n_estimators": 100,  # Número de árboles en el bosque
    "max_depth": None,  # Profundidad máxima del árbol
    "min_samples_split": 2,  # Mínimo de muestras para dividir un nodo
    "min_samples_leaf": 1,  # Mínimo de muestras en un nodo hoja
    "random_state": 42,  # Para reproducibilidad
    "bootstrap": True,  # Si se debe utilizar el muestreo con reemplazo
}
body = {"params": random_forest_params}
model_query = requests.post(url=url + "train/", json=body)
content = model_query.json()
print(content)
print("*" * 100)

print(requests.post("http://127.0.0.1:8000/predict/").json())
