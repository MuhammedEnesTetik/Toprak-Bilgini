from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from flask_cors import CORS  # type: ignore # CORS modülünü ekleyin

app = Flask(__name__)
CORS(app)  # CORS'u aktifleştirin

# Modeli yükle
model = joblib.load('')
label_encoder = joblib.load('')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = label_encoder.transform(df[column])
    
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    response = {
        'prediction': label_encoder.inverse_transform(prediction)[0],
        'probability': round(np.max(prediction_proba) * 100, 2)
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
