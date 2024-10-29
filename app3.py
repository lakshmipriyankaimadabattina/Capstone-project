# app.py

from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the pre-trained model and vectorizer for SMS Detection
sms_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
sms_model = pickle.load(open('model.pkl', 'rb'))

# Load the trained model for Website Phishing Detection
loaded_model = pickle.load(open('phishing.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict_sms', methods=['POST'])
def predict_sms():
    if request.method == 'POST':
        sms_text = request.form['sms_text']
        transformed_text = sms_vectorizer.transform([sms_text])
        prediction_sms = sms_model.predict(transformed_text)[0]
        return jsonify(result=prediction_sms)

@app.route('/predict_website', methods=['POST'])
def predict_website():
    if request.method == 'POST':
        url = request.form['url']
        result_website = loaded_model.predict([url])[0]
        return jsonify(result=result_website)

if __name__ == '__main__':
    app.run(debug=True)

