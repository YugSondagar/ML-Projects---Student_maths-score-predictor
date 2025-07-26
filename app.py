from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_datapoint', methods=['GET', 'POST'])  # Changed to match form action
def predict_datapoint():  # Renamed to match route
    if request.method == 'GET':
        return render_template('home.html')  # Directly serve the form page
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score')),
        )
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=round(results[0], 2))  # Rounded to 2 decimal places

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)