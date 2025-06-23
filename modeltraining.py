# aggregate_model.py
# Entrena un modelo de clasificación binaria a partir de estadísticas agregadas de equipos.
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

from flask import Flask, request, jsonify
from flask_cors import CORS
# …
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 1. Carga de datos agregados:
# CSV con columnas: Nombre_Equipo, Partidos jugados, Partidos ganados, Partidos perdidos, Goles por partido
df = pd.read_csv('team_stats.csv')

# 2. Construcción de pares para entrenamiento:
rows = []
for i, team_a in df.iterrows():
    for j, team_b in df.iterrows():
        if i >= j:
            continue
        # Características:
        diff_played = team_a['Partidos jugados'] - team_b['Partidos jugados']
        diff_won    = team_a['Partidos ganados'] - team_b['Partidos ganados']
        diff_lost   = team_a['Partidos perdidos'] - team_b['Partidos perdidos']
        diff_goals  = team_a['Goles por partido'] - team_b['Goles por partido']
        # Etiqueta: 1 si team_a es 'mejor' (más win ratio), 0 si team_b es mejor o igual
        ratio_a = team_a['Partidos ganados'] / team_a['Partidos jugados']
        ratio_b = team_b['Partidos ganados'] / team_b['Partidos jugados']
        label = 1 if ratio_a > ratio_b else 0
        rows.append({
            'diff_played': diff_played,
            'diff_won': diff_won,
            'diff_lost': diff_lost,
            'diff_goals': diff_goals,
            'label': label
        })
# DataFrame de entrenamiento
df_pairs = pd.DataFrame(rows)
X = df_pairs.drop('label', axis=1)
y = df_pairs['label']

# 3. Entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy en test: {accuracy_score(y_test, y_pred)*100:.2f}%")

# 4. Guardar modelo y estadísticas originales
data = {
    'model': model,
    'team_stats': df.set_index('Nombre_Equipo')
}
with open('aggregate_model.pkl', 'wb') as f:
    pickle.dump(data, f)
print("Modelo agregado entrenado y guardado en aggregate_model.pkl")
