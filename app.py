from flask import *
import cv2 as cv
import pickle

def classify_xray(img_path):
    classes = ['Normal', 'Pneumonia']
    image = cv.imread(img_path)
    image = cv.resize(image, (1000, 1000))
    image = image.reshape(1, 1000, 1000, 3)
    image = image / 255.0
    model = pickle.load(open('covid-19/model.pkl', 'rb'))
    prediction = model.predict(image)
    prediction = classes[prediction.argmax()]

    return prediction

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_path = 'static/uploads' + file.filename
        print(img_path)
        file.save(img_path)
        prediction = classify_xray(img_path)
        # return image with prediction and filename
        return render_template('index.html', prediction=prediction, img_path=img_path)
    
if __name__ == '__main__':
    app.run(debug=True)
