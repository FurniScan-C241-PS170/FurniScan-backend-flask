from flask import Flask, request, jsonify
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import io
import requests
from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)

# Function to download the model file
def download_model(url, local_path):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    with open(local_path, 'wb') as f:
        f.write(response.content)

# Function to preprocess and predict from image bytes
def tflite_detect_image(model_path, image_bytes, labels, min_conf=0.5):
    # Load the TensorFlow Lite model into memory
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Debug prints
    print("Input details:", input_details)
    print("Output details:", output_details)

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Load image and resize to expected shape [1xHxWx3]
    img = Image.open(io.BytesIO(image_bytes)).resize((width, height))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Ensure shape is [1, 224, 224, 3]

    # Set tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Extract the output probabilities
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Find the class with the highest probability
    max_score_index = np.argmax(output_data)
    max_score = output_data[max_score_index]

    # Check if the highest score meets the minimum confidence threshold
    if max_score > min_conf:
        detected_item = {
            'label': labels[max_score_index],
            'confidence': float(max_score),
        }
        return detected_item
    else:
        return {'label': 'No detection', 'confidence': 0.0}

# API endpoint to receive POST requests for prediction
@app.route('/predict', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        image_bytes = file.read()

        # Define your labels
        labels = [
            'kursi01', 'kursi02', 'kursi03', 'kursi04', 'kursi05', 'kursi06', 
            'kursi07', 'kursi08', 'kursi09', 'kursi10', 'kursi11', 'kursi12', 
            'meja01', 'sofa01'
        ]

        # Local path for the downloaded model
        local_model_path = 'new_furniture.tflite'

        # Download the model if it doesn't exist locally
        model_url = os.getenv('MODEL_URL')
        if not os.path.exists(local_model_path):
            download_model(model_url, local_model_path)

        # Perform detection on the uploaded image
        detection = tflite_detect_image(model_path=local_model_path,
                                        image_bytes=image_bytes,
                                        labels=labels,
                                        min_conf=0.5)  # Increase confidence threshold if necessary

        return jsonify(detection)

@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello world'

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=int(os.getenv("PORT",3000)))
