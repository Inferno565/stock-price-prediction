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
    logging.info(f"Received forecast request for {ticker}")
    result = predict(ticker)
    if result is None:
        return jsonify({'error': 'An error occurred during prediction'}), 500
    logging.info(f"Prediction result: {result}")
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
