import logging
from flask import Flask, render_template, request, jsonify
from main import predict

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Flask route to render the main page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to get predictions
@app.route('/predict', methods=['POST'])
def get_predictions():
    ticker = request.json['ticker']
    logging.info(f"Received prediction request for {ticker}")
    original, predictions = predict(ticker)
    logging.info(f"Prediction result: original length = {len(original)}, predictions length = {len(predictions)}")
    response = {
        'original': original,
        'predictions': predictions
    }
    logging.info(f"Sending response: {response}")
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
