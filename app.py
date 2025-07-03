from flask import Flask, jsonify, request
from sqlalchemy import create_engine
import joblib
import numpy as np
import pandas as pd
import datetime

host  = "35.241.178.229" 
port = 5432 
user= "postgres"
password= "postgres"
protocolo = "postgresql"
bbdd = "postgress"

churro = f"{protocolo}://{user}:{password}@{host}:{port}/{bbdd}"
engine = create_engine(churro)

model= joblib.load("model.pkl")

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/') #methods=['GET'])
def home():
    return "<h1>Bienvenido!</h1><p>Añade los datos de tu flor para predecir su tipo.</p>"

@app.route('/predict')
def predict():
        sepal_length = float(request.args["sepal_length"])
        sepal_width = float(request.args["sepal_width"])
        petal_length = float(request.args["petal_length"])
        petal_width = float(request.args["petal_width"])
        
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        prediction_id = model.predict(features)[0] # [0] para obtener el valor escalar
        
        iris_species_names = ['setosa', 'versicolor', 'virginica']
        predicted_species = iris_species_names[prediction_id]
        fecha_predict = str(datetime.datetime.now())
        input_str = str(features[0].tolist())
        
        dict = {
            "fecha prediccion": [fecha_predict], 
            "input": [input_str],
            "prediccion": [predicted_species]
        }

        df = pd.DataFrame(dict, index=None)
        df.to_sql("postgress",con=engine, if_exists="append")
        
        
        return jsonify({
            "message": "Prediccion realizada con exito",
            "Especie predicha": predicted_species})
        
        
@app.route('/historial')
def hist():
    query = """ SELECT * FROM postgress"""
    response = pd.read_sql(query, con=engine)
    return jsonify(response.to_dict(orient="records"))



app.run()