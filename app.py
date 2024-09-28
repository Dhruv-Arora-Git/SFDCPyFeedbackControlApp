from flask import Flask, request, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# test
@app.route('/')
def hello():
    return "Hari Bol"

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        # Try to get the 'text' from the incoming JSON request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']

        # Initialize the sentiment analyzer
        sia = SentimentIntensityAnalyzer()

        # Perform sentiment analysis
        sentiment = sia.polarity_scores(text)

        # Return the sentiment analysis result as JSON
        return jsonify(sentiment), 200

    except KeyError:
        # Handle the case where 'text' is not in the request JSON
        return jsonify({'error': 'KeyError: Missing text field in JSON data'}), 400

    except Exception as e:
        # Catch any other unforeseen errors
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8001)
