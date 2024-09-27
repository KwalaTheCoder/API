from flask import Flask, request, jsonify
import joblib
import numpy as np

# بارگذاری مدل
model = joblib.load('D:/data/House/housePrediction.sav')

# ساختن اپلیکیشن Flask
app = Flask(__name__)

# تعریف یک endpoint برای پیش‌بینی
@app.route('/predict', methods=['POST'])
def predict():
    # دریافت داده‌ها از درخواست POST
    data = request.json
    # تبدیل داده‌ها به فرمت مناسب برای مدل
    features = np.array([data['features']])
    
    # پیش‌بینی قیمت
    prediction = model.predict(features)
    
    # بازگشت نتیجه به صورت JSON
    return jsonify({'prediction': prediction[0]})

# اجرای اپلیکیشن
if __name__ == '__main__':
    app.run(debug=True)
