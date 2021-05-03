import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

classifier = pickle.load(open('classifier.pkl','rb'))

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    gender = request.form.get('gender')
    height = request.form.get('height')
    weight = request.form.get('weight')
    age = request.form.get('age')
    hypertension = request.form.get('hypertension')
    married = request.form.get('married')
    residence = request.form.get('residence')
    glucose = request.form.get('glucose')
    work = request.form.get('work')
    smoke = request.form.get('smoke')

    #calculating BMI
    bmi = int(weight)/(np.square(int(height)/100))

    x = [int(gender), int(age), int(hypertension), int(married), int(work), int(residence), int(glucose), bmi, int(smoke)]

    df = pd.DataFrame([x], columns=['gender', 'age', 'hypertension', 'ever_married',
                                    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])

    result = classifier.predict(df)

    if result == 0:
        res = 'You are Safe'
    else:
        res = 'There is a possibility of Stroke'

    return render_template('index.html',x=res)

if __name__ == "__main__":
    app.run(debug=True)