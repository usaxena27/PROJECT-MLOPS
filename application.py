from flask import Flask,render_template,request
import joblib
import numpy as np


app = Flask(__name__)

model_path = "artifacts/models/model.pk1"
scaler_path = "artifacts/processed/scaler.pk1"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route("/")
def home():
    return render_template("index.html", prediction = None)

@app.route("/predict", methods = ["POST"])
def predict():
    try:
        healthcare_costs = float(request.form["healthcare_costs"])
        tumor_size = float(request.form["tumor_size"])
        treatment_type = float(request.form["treatment_type"])
        diabetes = float(request.form["diabetes"])
        mortality_rate = float(request.form["mortality_rate"])
        
        input = np.array([[healthcare_costs, tumor_size, treatment_type, diabetes, mortality_rate]])

        scaled_input = scaler.transform(input)

        prediction = model.predict(scaled_input)[0]

        return render_template("index.html", prediction = prediction)
    except Exception as e:
        return str(e)
    
if __name__=="__main__":
        app.run(debug=True, host="0.0.0.0", port=5000)
        