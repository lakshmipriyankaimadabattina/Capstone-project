from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the pre-trained models and vectorizer
sms_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
sms_model = pickle.load(open('model.pkl', 'rb'))
website_model = pickle.load(open('phishing.pkl', 'rb'))
email_model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('unified_index.html')

@app.route('/predict_sms', methods=['POST'])
def predict_sms():
    if request.method == 'POST':
        sms_text = request.form['sms_text']
        transformed_text = sms_vectorizer.transform([sms_text])
        prediction = sms_model.predict(transformed_text)[0]
        return render_template('unified_index.html', prediction=prediction, sms_text=sms_text)

@app.route('/predict_website', methods=['POST'])
def predict_website():
    if request.method == 'POST':
        url = request.form['url']
        result = website_model.predict([url])[0]
        return render_template('unified_index.html', website_url=url, website_result=result)

@app.route('/predict_email', methods=['POST'])
def predict_email():
    if request.method == 'POST':
        email_text = request.form['email_text']
        transformed_text = email_vectorizer.transform([email_text])
        prediction = email_model.predict(transformed_text)[0]
        return render_template('unified_index.html', email_prediction=prediction, email_text=email_text)

if __name__ == '__main__':
    app.run(debug=True)
