import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

dataset = pd.read_csv('diabetes.csv')

dataset_X = dataset.iloc[:,[1, 2, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)


@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/check')
def check():
    return render_template('check.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( sc.transform(final_features) )

    if prediction == 1:
        pred = "Positif Diabetes, Konsultasi ke Dokter"
    elif prediction == 0:
        pred = "Negatif Diabetes, Konsultasi ke Dokter"
    output = pred

    return render_template('check.html', prediction_text='{}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)
