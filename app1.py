from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
# Load the pre-trained model and vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle the form submission
@app.route('/predictsms', methods=['POST'])
def predict():
    # Get the input text from the form
    sms_text = request.form['sms_text']

    # Transform the text using the vectorizer
    transformed_text = vectorizer.transform([sms_text])

    # Make the prediction using the model
    prediction = model.predict(transformed_text)[0]

    # Display the prediction result on the webpage
    return render_template('index.html', prediction=prediction, sms_text=sms_text)

if __name__ == '__main__':
    app.run(debug=True)
