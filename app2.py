from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
loaded_model = pickle.load(open('phishing.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predictwebsite', methods=['POST'])
def predict():
    if request.method == 'POST':
        url = request.form['url']
        result = loaded_model.predict([url])[0]
        return render_template('result.html', url=url, result=result)

if __name__ == '__main__':
    app.run(debug=True)
