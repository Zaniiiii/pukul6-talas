import os
from flask import Flask, jsonify
import json
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from datetime import datetime

app = Flask(__name__)

# Path ke direktori tempat hasil crawler disimpan
CRAWLER_JSON_DIR = '/Talas/CrawlerNews'

# Load the tokenizer from JSON
with open('/root/Talas/pukul6-talas/tokenizer.json', 'r') as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='/root/Talas/pukul6-talas/pro-antii_detection_lstm.tflite')
interpreter.allocate_tensors()

# Get input and output tensor information
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to classify a single text
def classify_text(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    max_len = 30  # Ensure the max length matches what was used during training
    padded_sequences = pad_sequences(sequences, maxlen=max_len)

    # Convert input data to float32
    padded_sequences = padded_sequences.astype('float32')

    # Set the tensor input with padded data
    interpreter.set_tensor(input_details[0]['index'], padded_sequences)

    # Run the interpreter to make a prediction
    interpreter.invoke()

    # Get prediction output tensor
    predictions_tflite = interpreter.get_tensor(output_details[0]['index'])

    # Interpret the prediction
    predicted_label = 'Anti' if predictions_tflite[0] > 0.5 else 'Pro'
    
    return float(predictions_tflite[0]), predicted_label

# Route untuk klasifikasi berita berdasarkan topik
@app.route('/api/classify_news/<topik>', methods=['GET'])
def classify_news(topik):
    try:
        # Format topik dan tanggal sesuai struktur penamaan file JSON dari crawler
        topik = topik.replace(" ", "-").replace("/", "-")
        date_str = datetime.now().strftime('%d%m%y')
        filename = f'scraped_news_{topik}_{date_str}.json'
        
        # Path lengkap ke file JSON
        filepath = os.path.join(CRAWLER_JSON_DIR, filename)

        # Cek apakah file JSON ada di direktori
        if not os.path.exists(filepath):
            return jsonify({'error': f'File {filename} not found'}), 404

        # Load the scraped news JSON file
        with open(filepath, 'r') as f:
            scraped_news = json.load(f)

        # Classify each news article
        classified_news = []
        for news_item in scraped_news:
            text = news_item.get('content', '')  # Assuming 'content' contains the text to classify
            confidence, label = classify_text(text)
            news_item['prediction'] = label
            news_item['confidence'] = confidence
            classified_news.append(news_item)

        # Return the classified news as JSON
        return jsonify(classified_news)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)
