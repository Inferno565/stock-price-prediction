from flask import Flask, render_template, request, jsonify
from main import predict

app = Flask(__name__)

# Flask route to render the main page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to get predictions
@app.route('/predict', methods=['POST'])
def get_predictions():
    ticker = request.json['ticker']
    predictions = predict(ticker)
    return jsonify(predictions.tolist())

if __name__ == "__main__":
    app.run(debug=True)
