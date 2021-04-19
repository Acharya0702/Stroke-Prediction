import numpy as np
from flask import Flask, request, jsonify, render_template
import sklearn.externals
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
model = joblib.load(open('Stroke.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('pred.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        f1 = float(request.form['f1'])
        f2 = float(request.form['f2'])
        f3 = float(request.form['f3'])
        f4 = float(request.form['f4'])
        f5 = float(request.form['f5'])
        f6 = float(request.form['f6'])
        f7 = float(request.form['f7'])
        feature_array = [f1,f2,f3,f4,f5,f6,f7]
        feature = np.array(feature_array).reshape(1,-1)

        prediction = model.predict(feature)

        dic = {'Stroke':0, 'No stroke':1}
        for key,value in dic.items():
            if value == prediction:
                x=key
        return render_template('pred.html', prediction='{} predicted.'.format(x))

if __name__ == '__stroke__':
    app.run(debug=True)