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

churro = "postgresql://postgres1:lHIPV0IBmc9PaFxbBLOg3qRIiWJqYxrh@dpg-d1j5lovdiees73cm37q0-a.frankfurt-postgres.render.com/postgresdb_n7m7"
engine = create_engine(churro)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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
        body{{font-family:Georgia,serif;margin:0;padding:20px;background:url('https://images.pexels.com/photos/67857/daisy-flower-spring-marguerite-67857.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1') no-repeat center center fixed;background-size:cover;color:#333;}}
        .container{{max-width:600px;margin:auto;background:rgba(255,255,255,0.95);padding:30px;border-radius:15px;box-shadow:0 8px 16px rgba(0,0,0,0.1);position:relative;overflow:hidden;}}
        .container::before{{content:'';position:absolute;top:0;left:0;width:100%;height:100%;background:url('https://www.transparenttextures.com/patterns/paper-fibers.png');opacity:0.2;z-index:-1;}}
        h1{{text-align:center;color:#4A7043;font-size:2.5em;margin-bottom:20px;text-shadow:1px 1px 2px rgba(0,0,0,0.1);}}
        form{{background:#fff;padding:25px;border-radius:10px;border:1px solid #e0e0e0;box-shadow:0 4px 8px rgba(0,0,0,0.05);}}
        label{{display:block;margin-bottom:10px;font-weight:bold;color:#6B4E31;font-size:1.1em;}}
        input[type="file"]{{width:calc(100% - 22px);padding:12px;margin-bottom:20px;border:1px solid #d4e4d4;border-radius:8px;background:#f8fff8;font-family:Georgia,serif;}}
        input[type="submit"]{{background:linear-gradient(45deg,#4CAF50,#81C784);color:white;padding:12px 20px;border:none;border-radius:25px;cursor:pointer;font-size:1.1em;transition:transform 0.2s,background 0.3s;}}
        input[type="submit"]:hover{{background:linear-gradient(45deg,#45a049,#66BB6A);transform:scale(1.05);}}
        .prediction-result{{margin-top:20px;padding:15px;background:#e6f3e6;border:1px solid #4CAF50;border-radius:10px;color:#font-family:Georgia,serif;}}
        .error-message{{margin-top:20px;padding:15px;background:#ffe6e6;border:1px solid #ff4c4c;border-radius:10px;color:#ff4c4c;font-style:italic;}}
        .image-display{{margin-top:20px;text-align:center;}}
        .image-display img{{max-width:100%;height:auto;border:2px solid #d4e4d4;border-radius:10px;box-shadow:0 4px 8px rgba(0,0,0,0.1);}}
        .flower-decor{{position:absolute;top:10px;right:10px;width:100px;height:100px;background:url('https://www.transparenttextures.com/patterns/flowers.png');opacity:0.3;z-index:-1;}}
    </style>
</head>
<body>
    <div class="container">
        <div class="flower-decor"></div>
        <h1>Predicción de Flor Iris</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <label for="plant_image">Adjunta una foto de tu flor:</label>
            <input type="file" id="plant_image" name="plant_image" accept="image/*" required>
            <input type="submit" value="Descubrir Flor">
        </form>
    </div>
</body>
</html> """

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
    app.run(host='0.0.0.0', port=5000, debug=True)