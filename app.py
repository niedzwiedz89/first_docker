from flask import Flask, request, jsonify
import joblib
import numpy as np
from flasgger import Swagger

# Tworzymy aplikację Flask
app = Flask(__name__)
swagger = Swagger(app)

# Trenujemy model przy starcie aplikacji
print("Wczytuję model...")

model = joblib.load('model.pkl')
#train_accuracy = model.score(iris.data, iris.target)

#print(f"Model gotowy! Accuracy: {train_accuracy}")

# Nazwy gatunków
SPECIES = ['setosa', 'versicolor', 'virginica']
@app.route('/health', methods=['GET'])
def health():
    """Endpoint do sprawdzenia czy serwis działa"""
    return jsonify({"status": "ok"})


@app.route('/predict', methods=['POST'])
def predict():
    """
    To jest opis dla Swaggera.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            sepal_length: {type: number, example: 5.1}
            sepal_width: {type: number, example: 3.5}
            petal_length: {type: number, example: 1.4}
            petal_width: {type: number, example: 0.2}
    responses:
      200:
        description: Wynik predykcji
    """
    # Pobierz dane z requestu
    data = request.get_json()

    # Przygotuj features dla modelu
    features = np.array([[
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]])

    # Predykcja
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features).max()

    # Zwróć wynik
    return jsonify({
        "species": SPECIES[prediction],
        "probability": round(float(probability), 3)
    })


@app.route('/', methods=['GET'])
def home():
    """Strona główna z instrukcją"""
    return jsonify({
        "message": "Iris Classifier API",
        "endpoints": {
            "GET /health": "Sprawdź status",
            "POST /predict": "Wyślij dane irysa, otrzymaj predykcję"
        },
        "example_input": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    })

@app.route('/form',methods=['GET'])
def form():
    """Prosty formularz HTML do predykcjie używający endpointa /predict"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Klasyfikator irysów</title>
    </head>
    <body>
        <h1> Klasyfikator irysów </h1>
        <input id = "sl" type="number" value="5.1" placeholder="sepal_length"><br>
        <input id = "sw" type="number" value="3.1" placeholder="sepal_width"><br>
        <input id = "pl" type="number" value="1.4" placeholder="petal_length"><br>
        <input id =  "pw" type="number" value="0.1" placeholder="petal_width"><br>
        <button onclick="predict()">Predykcja</button>
        <h2 id= "result"></h2>
        
        <script>
            async function predict(){
                const data = {
                 sepal_length: parseFloat(document.getElementById('sl').value),
                 sepal_width: parseFloat(document.getElementById('sw').value),
                 petal_length: parseFloat(document.getElementById('pl').value),
                 petal_width: parseFloat(document.getElementById('pw').value)
                }
                
                const res = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
                })
                
                const result = await res.json()
                
                document.getElementById('result').innerText = 'Gatunetk: ' + result.species + ' (Prawdopodobieństwo: ' + (result.probability*100) + '%)'
                
            }
        </script>
    </body>
    </html>
    '''

@app.route('/classic-form', methods=['GET', 'POST'])
def classic_form():
    result_html = ""

    # Jeśli użytkownik kliknął "Wyślij" (metoda POST)
    if request.method == 'POST':
        # 1. Pobieramy dane z formularza (zamiast JSON)
        # UWAGA: Dane z formularza przychodzą jako napisy, trzeba rzutować na float
        try:
            features = np.array([[
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]])

            # 2. Robimy predykcję (tak samo jak wcześniej)
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features).max()

            # 3. Przygotowujemy wynik do wyświetlenia
            wynik_tekst = f"{SPECIES[prediction]} ({round(probability * 100, 1)}%)"
            result_html = f"<h3>Wynik: {wynik_tekst}</h3>"

        except ValueError:
            result_html = "<h3 style='color:red'>Błąd: Wpisz poprawne liczby!</h3>"

    # Zwracamy kod HTML (formularz + ewentualny wynik)
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Klasyfikator Bez JS</title>
    </head>
    <body>
        <h1>Klasyfikator irysów (Wersja Classic)</h1>

        <form method="POST">
            <label>Sepal Length:</label>
            <input name="sepal_length" type="number" step="0.1" required><br>

            <label>Sepal Width:</label>
            <input name="sepal_width" type="number" step="0.1" required><br>

            <label>Petal Length:</label>
            <input name="petal_length" type="number" step="0.1" required><br>

            <label>Petal Width:</label>
            <input name="petal_width" type="number" step="0.1" required><br>
            <br>
            <button type="submit">Sprawdź gatunek</button>
        </form>

        {result_html}

    </body>
    </html>
    '''


@app.route('/info', methods=['GET'])
def model_info():
    """Zwraca informacje o modelu: typ, parametry i dokładność"""
    return jsonify({
        "model_type": type(model).__name__,  # Np. "RandomForestClassifier"
        "model_params": model.get_params(),  # Słownik wszystkich parametrów
        "train_accuracy": train_accuracy     # Obliczona wcześniej dokładność
    })


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Endpoint do predykcji dla wielu irysów na raz.
    """
    # 1. Pobieramy cały JSON
    request_data = request.get_json()

    # Upewniamy się, że klucz "data" istnieje
    if 'data' not in request_data:
        return jsonify({"error": "Missing 'data' key"}), 400

    input_list = request_data['data']

    # 2. Przekształcamy listę słowników na macierz (listę list)
    # Scikit-learn potrzebuje formatu: [[5.1, 3.5...], [6.0, 2.7...]]
    features = []
    for item in input_list:
        features.append([
            item['sepal_length'],
            item['sepal_width'],
            item['petal_length'],
            item['petal_width']
        ])

    # 3. Wykonujemy predykcję DLA WSZYSTKICH naraz (Vectorization)
    # To jest ten moment, gdzie model dostaje np. 50 wierszy i zwraca 50 wyników
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    # 4. Pakujemy wyniki z powrotem do ładnego JSON-a
    results = []
    for i in range(len(predictions)):
        results.append({
            "input_index": i,
            "species": SPECIES[predictions[i]],
            "probability": round(float(probabilities[i].max()), 3)
        })

    return jsonify(results)

@app.route('/predict_batch_zajecia', methods=['POST'])
def predict_batch_zajecia():
    data = request.get_json()

    features = np.array([
        [obs['sepal_length'], obs['sepal_width'], obs['petal_length'], obs['petal_width']]
        for obs in data['data']
    ])

    predictions = model.predict(features)
    probabilitys = model.predict_proba(features).max(axis=1)

    results = [
        {"species": SPECIES[pred], "probability": round(float(prob), 3)}
        for pred, prob in zip(predictions, probabilitys)
    ]

    return jsonify({"predictions": results})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

