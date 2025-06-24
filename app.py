from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Carga del modelo y estadísticas (ya entrenado)
with open('aggregate_model.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
team_stats = data['team_stats']  # DataFrame indexado por Nombre_Equipo

# Ruta del archivo de contadores
COUNTER_FILE = 'counters.json'

# Carga o inicializa contadores
def load_counters():
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_counters(counters):
    with open(COUNTER_FILE, 'w') as f:
        json.dump(counters, f, indent=2)

def update_counter(team1, team2):
    counters = load_counters()
    key = '_vs_'.join(sorted([team1, team2]))
    counters[key] = counters.get(key, 0) + 1
    save_counters(counters)

def predict_probs(team1: str, team2: str):
    if team1 not in team_stats.index or team2 not in team_stats.index:
        raise ValueError(f"Equipo desconocido: {team1} o {team2}")

    a = team_stats.loc[team1]
    b = team_stats.loc[team2]
    feats = {
        'diff_played': a['Partidos jugados'] - b['Partidos jugados'],
        'diff_won':    a['Partidos ganados'] - b['Partidos ganados'],
        'diff_lost':   a['Partidos perdidos'] - b['Partidos perdidos'],
        'diff_goals':  a['Goles por partido'] - b['Goles por partido'],
    }
    X_df = pd.DataFrame([feats])
    proba = model.predict_proba(X_df)[0]
    p1 = round(proba[1] * 100, 2)
    p2 = round(proba[0] * 100, 2)
    rec = team1 if p1 > p2 else team2
    return {'team1': p1, 'team2': p2, 'recommendation': rec}

@app.route('/predict', methods=['POST'])
def metodo_predict():
    try:
        datos = request.get_json()
        t1 = datos.get('team1')
        t2 = datos.get('team2')
        if not t1 or not t2:
            return jsonify(error="Debe incluir 'team1' y 'team2' en el JSON"), 400

        res = predict_probs(t1, t2)

        # Actualizar contador
        update_counter(t1, t2)

        output = {
            t1: res['team1'],
            t2: res['team2'],
            'recommendation': res['recommendation']
        }
        return jsonify(output)

    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error="Error interno: " + str(e)), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
