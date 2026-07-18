# Importing essential libraries
from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf


app = Flask(__name__)
app.secret_key = 'O.\x89\xcc\xa0>\x96\xf7\x871\xa2\xe6\x9a\xe4\x14\x91\x0e\xe5)\xd9'

# Load the Random Forest CLassifier model
filename = 'Models/diabetes_r.pkl'
classifier = joblib.load(open(filename, 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            Gender = int(request.form['gender'])
            FPG = int(request.form['FPG'])
            Chol = int(request.form['Chol'])
            Tri = int(request.form['Tri'])
            HDL = int(request.form['HDL'])
            LDL = int(request.form['LDL'])
            ALT = int(request.form['ALT'])
            Creatinine = float(request.form['Creatinine'])

            data = np.array([[age, Gender, FPG, Chol, Tri, HDL, LDL, ALT, Creatinine]])
            my_prediction = classifier.predict(data)

            return render_template('d_result.html', prediction=my_prediction)
        except ValueError:
            flash(
                'Invalid input. Please fill in the form with appropriate values', 'info')
            return redirect(url_for('diabetes'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')


if __name__ == '__main__':
    app.run(debug=True,port=8080)
