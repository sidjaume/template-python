from flask import Flask, request, jsonify, render_template
import pandas as pd
import sqlite3
import numpy as np
from sklearn.datasets import load_iris
import pickle as pkl
import psycopg2

app = Flask(__name__)

from sqlalchemy import create_engine

# Crea una conexión a la base de datos
engine = create_engine('postgresql://fl0user:9aIMLsfioV0r@ep-odd-cell-06464893.eu-central-1.aws.neon.tech:5432/postgres?sslmode=require')


@app.route('/', methods=["GET"])
def form():
       
       return render_template('form.html')

@app.route('/v0/get_predict', methods=["POST"])
def get_predict():
    def error():
        return "DATA ERROR"
    
    feat1 = request.form.get('feat1', None)
    feat2 = request.form.get('feat2', None)
    feat3 = request.form.get('feat3', None)
    feat4 = request.form.get('feat4', None)
    
    
    if feat1 is None or feat2 is None or feat3 is None or feat4 is None:
        return error()

    try:
        feat1 = float(feat1)
        feat2 = float(feat2)
        feat3 = float(feat3)
        feat4 = float(feat4)
                
        if (feat1 < 0) or (feat2 < 0) or (feat3 < 0) or (feat4 < 0):
            return "Data must be positive."
        
    except ValueError:
        return error()
    
    if feat1 > 6.0:
        feat1 = 1.
    else:
        feat1 = 0.

    raw = np.array([[feat1, feat2, feat3, feat4]])

    with open('scaler.pkl', 'rb') as archivo_entrada:
        scaler = pkl.load(archivo_entrada)
    numeros = scaler.transform(raw)

    with open('iris_model.pkl', 'rb') as archivo_entrada:
        modelo_importado = pkl.load(archivo_entrada)
    data = load_iris()
    prediction = data.target_names[modelo_importado.predict(numeros)[0]]

    from datetime import datetime

    time = str(datetime.now())
    nums = str(raw[0]).split()
    nums = ','.join(nums)
    cols = {
        'measures' : nums,
        'type' : prediction,
        'time' : time
        }
    
    df = pd.DataFrame(cols, index=[int(datetime.now().timestamp())])
    df.to_sql(name="predictions",if_exists='append',con=engine)
       return jsonify(prediction)


@app.route('/v0/get_logs', methods=["GET"])
def get_logs():
    return jsonify(pd.read_sql_query("select * from predictions", con = engine).to_dict("records"))

@app.route('/v0/del_logs', methods=["DELETE"])
def del_logs():    

    conn = sqlite3.connect('flowers.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return 'Table deleted succesfully!'

if __name__ == '__main__':
    app.run(debug=True, port = 18473)
