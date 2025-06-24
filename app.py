from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
import csv

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

COUNTER_FILE = 'counters.csv'

# --- Utility functions for the counter file ---

def ensure_counter_file():
    """Make sure the counter file exists; if not, create with header."""
    if not os.path.isfile(COUNTER_FILE):
        with open(COUNTER_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # no header, just rows of team1,team2,count
            pass  # creates empty file

def read_counters():
    """Read all counters into a dict keyed by (team1, team2)."""
    counters = {}
    with open(COUNTER_FILE, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            t1, t2, cnt = row
            try:
                counters[(t1, t2)] = int(cnt)
            except ValueError:
                continue
    return counters

def write_counters(counters):
    """Overwrite the counter file with the given dict."""
    with open(COUNTER_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for (t1, t2), cnt in counters.items():
            writer.writerow([t1, t2, cnt])

def increment_match_counter(team1, team2):
    """
    Increment the counter for this specific pairing.
    Returns the new counter value.
    """
    ensure_counter_file()
    counters = read_counters()
    key = (team1, team2)
    if key in counters:
        counters[key] += 1
    else:
        counters[key] = 1
    write_counters(counters)
    return counters[key]

def get_match_counter(team1, team2):
    """Return the current counter for this pairing (0 if not found)."""
    if not os.path.isfile(COUNTER_FILE):
        return 0
    counters = read_counters()
    return counters.get((team1, team2), 0)

# --- Load your model/statistics as before ---

with open('aggregate_model.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
team_stats = data['team_stats']  # DataFrame indexed by Nombre_Equipo

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

# --- Modified /predict to include the counter logic ---

@app.route('/predict', methods=['POST'])
def metodo_predict():
    try:
        datos = request.get_json()
        t1 = datos.get('team1')
        t2 = datos.get('team2')
        if not t1 or not t2:
            return jsonify(error="Debe incluir 'team1' y 'team2' en el JSON"), 400

        # 1) increment the counter for this match
        new_count = increment_match_counter(t1, t2)

        # 2) run the prediction as before
        res = predict_probs(t1, t2)
        output = {
            t1: res['team1'],
            t2: res['team2'],
            'recommendation': res['recommendation'],
            'match_count': new_count      # <-- include the updated counter
        }
        return jsonify(output)

    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error="Error interno: " + str(e)), 500

# --- New endpoint /contador ---

@app.route('/contador', methods=['GET'])
def obtener_contador():
    """
    Usage: GET /contador?team1=Pachuca&team2=México
    Returns JSON: { "team1": <name>, "team2": <name>, "count": <int> }
    """
    t1 = request.args.get('team1')
    t2 = request.args.get('team2')
    if not t1 or not t2:
        return jsonify(error="Debe pasar 'team1' y 'team2' como parámetros"), 400

    cnt = get_match_counter(t1, t2)
    return jsonify({
        'team1': t1,
        'team2': t2,
        'count': cnt
    })

# --- Run the app as before ---

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
