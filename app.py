from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Carga del modelo y estadísticas (ya entrenado)
with open('aggregate_model.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
team_stats = data['team_stats']  # DataFrame indexado por Nombre_Equipo

def predict_probs(team1: str, team2: str):
    if team1 not in team_stats.index or team2 not in team_stats.index:
        raise ValueError(f"Equipo desconocido: {team1} o {team2}")

    a = team_stats.loc[team1]
    b = team_stats.loc[team2]
    # diferencias de estadísticas
    feats = {
        'diff_played': a['Partidos jugados'] - b['Partidos jugados'],
        'diff_won':    a['Partidos ganados'] - b['Partidos ganados'],
        'diff_lost':   a['Partidos perdidos'] - b['Partidos perdidos'],
        'diff_goals':  a['Goles por partido'] - b['Goles por partido'],
    }
    X_df = pd.DataFrame([feats])
    proba = model.predict_proba(X_df)[0]
    p1 = round(proba[1] * 100, 2)  # prob de team1 “mejor”
    p2 = round(proba[0] * 100, 2)  # prob de team2 “mejor”
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
        # renombramos para que la salida use los nombres reales
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
    # Render asignará su propio puerto via env var PORT
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

"""
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
"""
