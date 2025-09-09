from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            gender = int(request.form['gender'])
            hormone = int(request.form['hormone'])
            fhistory = int(request.form['fhistory'])
            weight = int(request.form['weight'])
            activity = int(request.form['activity'])
            smoking = int(request.form['smoking'])
            medcondition = int(request.form['medcondition'])
            fractures = int(request.form['fractures'])

            input_features = np.array([[age, gender, hormone, fhistory, 
                                        weight,  activity, 
                                        smoking, medcondition, fractures]])
            
            prediction = model.predict(input_features)[0]

            result = "High Risk" if prediction == 1 else "Low Risk"
            return render_template('index.html', prediction_text=f'Osteoporosis Risk: {result}')

        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
