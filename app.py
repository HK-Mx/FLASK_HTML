import os
from flask import Flask, jsonify, request, render_template_string
from sqlalchemy import create_engine
import joblib
import numpy as np
import pandas as pd
import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import json
import base64

load_dotenv()

churro = "postgresql://postgres:postgres@34.38.195.15/postgres"
engine = create_engine(churro)

GEMINI_API_KEY = os.environ("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY no encontrada en las variables de entorno.")

genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Usamos el modelo de visión para imágenes

model = joblib.load("model.pkl")

app = Flask(__name__)

HTML_FORM = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predicción de Flor Iris</title>
    <style>
        body {{font-family: Arial, sans-serif; margin: 20px; }}
        form {{background-color: #f4f4f4; padding: 20px; border-radius: 8px; max-width: 500px; margin: auto; }}
        label {{display: block; margin-bottom: 8px; font-weight: bold; }}
        input[type="file"] {{width: calc(100% - 22px); padding: 10px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 4px; }}
        input[type="submit"] {{background-color: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }}
        input[type="submit"]:hover {{ background-color: #45a049; }}
        .prediction-result {{margin-top: 20px; padding: 15px; background-color: #e9ffe9; border: 1px solid #4CAF50; border-radius: 8px; }}
        .error-message {{margin-top: 20px; padding: 15px; background-color: #ffe9e9; border: 1px solid #ff4c4c; border-radius: 8px; color: #ff4c4c; }}
        .image-display {{ margin-top: 20px; text-align: center; }}
        .image-display img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Predicción de Flor Iris</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <label for="plant_image">Adjuntar foto de la planta:</label>
        <input type="file" id="plant_image" name="plant_image" accept="image/*" required><br>
        <input type="submit" value="Predecir">
    </form>

    {image_display}
    {prediction_result}
</body>
</html>
"""

@app.route('/')
def home():
    return "<h1>¡Bienvenido!</h1><p>Adjunta la foto de tu flor para predecir su tipo. Dirígete a /predict</p>"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_result_html = ""
    image_display_html = ""

    if request.method == 'POST':
        image_file = request.files['plant_image']
        image_data = image_file.read()

        # Encode de la imagen a base64 para HTML
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        image_display_html = f"""
        <div class="image-display">
            <h2>Imagen Subida:</h2>
            <img src="data:{image_file.content_type};base64,{image_b64}" alt="Imagen de la planta">
        </div>
        """
        
        prompt_parts_measurements = [
            "Analiza la imagen de esta flor y proporciona las siguientes medidas en formato JSON. Si algún valor no es discernible, estima un valor razonable. ¡IMPORTANTE! Siempre devuelve el JSON en el mismo formato, incluso si un valor es estimado o faltante. El JSON debe tener las claves 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', y los valores deben ser números flotantes en centímetros. Por ejemplo: {'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2}.",
            {
                "mime_type": image_file.content_type,
                "data": image_data
            }
        ]
        
        response_measurements = gemini_model.generate_content(prompt_parts_measurements)
        
        json_start = response_measurements.text.find('{')
        json_end = response_measurements.text.rfind('}') + 1
        json_string = response_measurements.text[json_start:json_end]
        
        features_dict = json.loads(json_string)
        
        sepal_length = float(features_dict['sepal_length'])
        sepal_width = float(features_dict['sepal_width'])
        petal_length = float(features_dict['petal_length'])
        petal_width = float(features_dict['petal_width'])

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        prediction_id = model.predict(features)[0]
        iris_species_names = ['setosa', 'versicolor', 'virginica']
        
        predicted_species = iris_species_names[prediction_id]
        fecha_predict = str(datetime.datetime.now())
        
        input_str = str(features[0].tolist())

        data_to_store = {
            "fecha prediccion": [fecha_predict],
            "input": [input_str],
            "prediccion": [predicted_species]
        }
        df = pd.DataFrame(data_to_store)

        
        df.to_sql("postgres", con=engine, if_exists="append", index=False)
            

        prompt_explanation = [
            f"La imagen que me proporcionaste es de una flor de iris. Basado en las características, ha sido predicha como '{predicted_species}'. Dame una explicación concisa y breve de por qué esta flor podría ser de esta especie, mencionando alguna característica clave que se vea en la imagen. Mantén la explicación en español.",
            {
                "mime_type": image_file.content_type,
                "data": image_data
            }
        ]
        response_explanation = gemini_model.generate_content(prompt_explanation)
        explanation_text = response_explanation.text
        
        prediction_result_html = f"""
        <div class="prediction-result">
            <h2>Resultado de la Predicción:</h2>
            <p><strong>Especie predicha:</strong> {predicted_species}</p>
            <p><strong>Explicación (por IA):</strong> {explanation_text}</p>
            <p><strong>Longitud Sépalo (estimado por IA):</strong> {sepal_length:.2f} cm</p>
            <p><strong>Ancho Sépalo (estimado por IA):</strong> {sepal_width:.2f} cm</p>
            <p><strong>Longitud Pétalo (estimado por IA):</strong> {petal_length:.2f} cm</p>
            <p><strong>Ancho Pétalo (estimado por IA):</strong> {petal_width:.2f} cm</p>
            <p><strong>Fecha de Predicción:</strong> {fecha_predict}</p>
        </div>
        """
        return HTML_FORM.format(image_display=image_display_html, prediction_result=prediction_result_html)

    return HTML_FORM.format(image_display=image_display_html, prediction_result=prediction_result_html)

@app.route('/historial')
def hist():
    with engine.connect() as connection:
        query = """ SELECT * FROM postgres"""
        response = pd.read_sql(query, con=connection)
    
    return jsonify(response.to_dict(orient="records"))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)