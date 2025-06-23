# test_model.py
# Script para probar el modelo entrenado en aggregate_model.pkl
import pickle
import numpy as np
import sys
import pandas as pd

# 1. Cargar el modelo y las estadísticas
data = pickle.load(open('aggregate_model.pkl', 'rb'))
model = data['model']
team_stats = data['team_stats']  # DataFrame indexado por Nombre_Equipo

# 2. Función de predicción def predict(team1, team2):
def predict(team1: str, team2: str):
    if team1 not in team_stats.index or team2 not in team_stats.index:
        raise ValueError(f"Equipo desconocido: {team1} o {team2}")

    a = team_stats.loc[team1]
    b = team_stats.loc[team2]
    # Crear vector de diferencias
    diff_played = a['Partidos jugados'] - b['Partidos jugados']
    diff_won    = a['Partidos ganados'] - b['Partidos ganados']
    diff_lost   = a['Partidos perdidos'] - b['Partidos perdidos']
    diff_goals  = a['Goles por partido'] - b['Goles por partido']
    feature_names = ['diff_played', 'diff_won', 'diff_lost', 'diff_goals']

    # Construyes un DataFrame con exactamente esos nombres:
    X_df = pd.DataFrame([{
        'diff_played': diff_played,
        'diff_won':    diff_won,
        'diff_lost':   diff_lost,
        'diff_goals':  diff_goals
    }])

    proba = model.predict_proba(X_df)[0]
    p1 = proba[1] * 100  # probabilidad de que team1 sea "mejor"
    p2 = proba[0] * 100  # probabilidad de que team2 sea mejor
    return round(p1, 2), round(p2, 2)

# 3. Uso desde línea de comandos
def main():
    if len(sys.argv) != 3:
        print("Uso: python test_model.py "
              "<Equipo1> <Equipo2>")
        sys.exit(1)
    team1, team2 = sys.argv[1], sys.argv[2]
    try:
        p1, p2 = predict(team1, team2)
        recommended = team1 if p1 > p2 else team2
        print(f"{team1}: {p1}% vs {team2}: {p2}%")
        print(f"Recomendación: apostar por {recommended}")
    except ValueError as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main()
