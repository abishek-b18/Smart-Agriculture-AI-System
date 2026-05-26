from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load Crop Recommendation Model
crop_model = pickle.load(open('models/crop_model.pkl', 'rb'))

# Load Disease Detection Model
disease_model = load_model('models/disease_model.h5')

disease_classes = [
    'Healthy',
    'Bacterial Spot',
    'Early Blight',
    'Late Blight'
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/crop_prediction')
def crop_prediction():
    return render_template('crop_prediction.html')


@app.route('/predict_crop', methods=['POST'])
def predict_crop():

    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    features = np.array([[N,P,K,temperature,humidity,ph,rainfall]])

    prediction = crop_model.predict(features)

    result = prediction[0]

    return render_template(
        'crop_prediction.html',
        prediction=result
    )


@app.route('/disease_detection')
def disease_detection():

    return render_template(
        'disease_detection.html'
    )


@app.route('/predict_disease', methods=['POST'])
def predict_disease():

    file = request.files['image']

    filepath = os.path.join(
        "static/uploads",
        file.filename
    )

    file.save(filepath)

    img = image.load_img(
        filepath,
        target_size=(128,128)
    )

    img = image.img_to_array(img)

    img = np.expand_dims(
        img,
        axis=0
    )

    img = img /255

    prediction = disease_model.predict(img)

    disease = disease_classes[
        np.argmax(prediction)
    ]

    return render_template(
        'disease_detection.html',
        disease=disease,
        image=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)