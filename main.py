from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
from torchvision import transforms
import torch
import onnxruntime as rt
import numpy as np


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

MODEL_PATH = 'trained_models/resnet50.onnx'
model = rt.InferenceSession(MODEL_PATH)
INPUT_SIZE = 224
id2label = {2: 'Benign keratosis-like lesions ',
            4: 'Melanocytic nevi',
            3: 'Dermatofibroma',
            5: 'Melanoma',
            6: 'Vascular lesions',
            1: 'Basal cell carcinoma',
            0: 'Actinic keratoses'}

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        file = request.files['image']
        img_bytes = file.read()

        img = Image.open(io.BytesIO(img_bytes))
        preprocess = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7630392, 0.5456477, 0.57004845], std=[0.1409286, 0.15261266, 0.16997074]),
        ])

        img = preprocess(img)
        img = img.unsqueeze(0)
        prediction = onnx_predict(img, model)

        prediction_labels = {}
        for i in range(len(prediction)):
            prediction_labels[id2label[i]] = round(prediction[i] * 100, 2)

        return jsonify(prediction_labels)
    
def onnx_predict(img, rt_session):
    model_inputs = {"input.1": img.cpu().numpy()}
    ort_outs = rt_session.run(None, model_inputs)

    prediction = torch.softmax(torch.tensor(ort_outs[0]), dim=1)
    # convert prediction to list
    prediction = prediction.squeeze(0).tolist()

    return prediction


if __name__ == '__main__':
    app.run(debug=True)

    

if __name__ == '__main__':
    app.run()