import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained AdaBoost model
MODEL_PATH = 'adaboost.pkl'
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"Warning: {MODEL_PATH} not found. Please place it in the project root directory.")
except Exception as e:
    print(f"Error loading model: {e}")

# HTML Template with an attractive, responsive modern UI
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance & Admission Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-gradient: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border: #cbd5e1;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            padding: 40px 20px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            background: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .header {
            background: var(--primary);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .header p {
            font-size: 14px;
            opacity: 0.9;
        }

        .form-container {
            padding: 30px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-muted);
            margin-bottom: 6px;
        }

        input, select {
            padding: 10px 14px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            color: var(--text-main);
            transition: all 0.2s ease;
            background: #f8fafc;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            background: #ffffff;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }

        .btn-submit {
            margin-top: 30px;
            width: 100%;
            background: var(--primary);
            color: white;
            border: none;
            padding: 14px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.2s ease;
        }

        .btn-submit:hover {
            background: var(--primary-hover);
        }

        .result-box {
            margin-top: 25px;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            animation: fadeIn 0.4s ease-in-out;
        }

        .success {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
        }

        .error {
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-5px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AdaBoost Prediction Portal</h1>
            <p>Enter student profile metrics to evaluate model prediction</p>
        </div>
        <div class="form-container">
            <form method="POST">
                <div class="grid">
                    <div class="form-group">
                        <label>Age</label>
                        <input type="number" name="age" required value="{{ form_data.get('age', 18) }}">
                    </div>
                    <div class="form-group">
                        <label>Gender</label>
                        <select name="gender">
                            <option value="Male" {% if form_data.get('gender') == 'Male' %}selected{% endif %}>Male</option>
                            <option value="Female" {% if form_data.get('gender') == 'Female' %}selected{% endif %}>Female</option>
                            <option value="Other" {% if form_data.get('gender') == 'Other' %}selected{% endif %}>Other</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>State</label>
                        <input type="text" name="state" required value="{{ form_data.get('state', 'California') }}">
                    </div>
                    <div class="form-group">
                        <label>Family Income ($)</label>
                        <input type="number" step="any" name="family_income" required value="{{ form_data.get('family_income', 75000) }}">
                    </div>
                    <div class="form-group">
                        <label>High School GPA</label>
                        <input type="number" step="0.01" name="high_school_gpa" required value="{{ form_data.get('high_school_gpa', 3.5) }}">
                    </div>
                    <div class="form-group">
                        <label>SAT Score</label>
                        <input type="number" name="sat_score" required value="{{ form_data.get('sat_score', 1200) }}">
                    </div>
                    <div class="form-group">
                        <label>ACT Score</label>
                        <input type="number" name="act_score" required value="{{ form_data.get('act_score', 25) }}">
                    </div>
                    <div class="form-group">
                        <label>Attendance Rate (%)</label>
                        <input type="number" step="0.01" name="attendance_rate" required value="{{ form_data.get('attendance_rate', 95.0) }}">
                    </div>
                    <div class="form-group">
                        <label>AP Courses Count</label>
                        <input type="number" name="ap_courses" required value="{{ form_data.get('ap_courses', 3) }}">
                    </div>
                    <div class="form-group">
                        <label>Extracurricular Count</label>
                        <input type="number" name="extracurricular_count" required value="{{ form_data.get('extracurricular_count', 2) }}">
                    </div>
                    <div class="form-group">
                        <label>Volunteer Hours</label>
                        <input type="number" name="volunteer_hours" required value="{{ form_data.get('volunteer_hours', 50) }}">
                    </div>
                    <div class="form-group">
                        <label>Leadership Positions</label>
                        <input type="number" name="leadership_positions" required value="{{ form_data.get('leadership_positions', 1) }}">
                    </div>
                    <div class="form-group">
                        <label>Coding Projects</label>
                        <input type="number" name="coding_projects" required value="{{ form_data.get('coding_projects', 2) }}">
                    </div>
                    <div class="form-group">
                        <label>Social Media Hours / Day</label>
                        <input type="number" step="0.1" name="social_media_hours" required value="{{ form_data.get('social_media_hours', 2.5) }}">
                    </div>
                    <div class="form-group">
                        <label>Online Certifications</label>
                        <input type="number" name="online_certifications" required value="{{ form_data.get('online_certifications', 1) }}">
                    </div>
                    <div class="form-group">
                        <label>Essay Score</label>
                        <input type="number" step="0.1" name="essay_score" required value="{{ form_data.get('essay_score', 85.0) }}">
                    </div>
                    <div class="form-group">
                        <label>Recommendation Score</label>
                        <input type="number" step="0.1" name="recommendation_score" required value="{{ form_data.get('recommendation_score', 90.0) }}">
                    </div>
                    <div class="form-group">
                        <label>Interview Score</label>
                        <input type="number" step="0.1" name="interview_score" required value="{{ form_data.get('interview_score', 88.0) }}">
                    </div>
                </div>
                <button type="submit" class="btn-submit">Run Prediction</button>
            </form>

            {% if prediction is not none %}
                <div class="result-box {% if error %}error{% else %}success{% endif %}">
                    {% if error %}
                        {{ prediction }}
                    {% else %}
                        Prediction Result: <span>{{ prediction }}</span>
                    {% endif %}
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = False
    form_data = {}

    if request.method == 'POST':
        if model is None:
            error = True
            prediction = "Error: Model not loaded. Ensure 'adaboost.pkl' is in the project directory."
        else:
            try:
                # Extract features in the exact order expected by the model
                form_data = request.form
                features = [
                    float(form_data['age']),
                    form_data['gender'],
                    form_data['state'],
                    float(form_data['family_income']),
                    float(form_data['high_school_gpa']),
                    float(form_data['sat_score']),
                    float(form_data['act_score']),
                    float(form_data['attendance_rate']),
                    int(form_data['ap_courses']),
                    int(form_data['extracurricular_count']),
                    int(form_data['volunteer_hours']),
                    int(form_data['leadership_positions']),
                    int(form_data['coding_projects']),
                    float(form_data['social_media_hours']),
                    int(form_data['online_certifications']),
                    float(form_data['essay_score']),
                    float(form_data['recommendation_score']),
                    float(form_data['interview_score'])
                ]

                # Convert into a pandas DataFrame matching feature_names_in_ from the pickled model
                feature_names = [
                    'age', 'gender', 'state', 'family_income', 'high_school_gpa', 
                    'sat_score', 'act_score', 'attendance_rate', 'ap_courses', 
                    'extracurricular_count', 'volunteer_hours', 'leadership_positions', 
                    'coding_projects', 'social_media_hours', 'online_certifications', 
                    'essay_score', 'recommendation_score', 'interview_score'
                ]
                
                input_df = pd.DataFrame([features], columns=feature_names)

                # Make prediction
                pred = model.predict(input_df)
                prediction = str(pred[0])
            
            except Exception as e:
                error = True
                prediction = f"Prediction Error: {str(e)}"

    return render_template_string(TEMPLATE, prediction=prediction, error=error, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
