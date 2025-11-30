from flask import Flask, render_template, request
from joblib import load
import numpy as np
import os
import mlflow
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn

app = Flask(__name__)

# ==================== Paths ====================
MODEL_PATH = os.path.join("models", "patient_history.joblib")
SCALER_PATH = os.path.join("models", "scaler.joblib")

# ==================== Load Model & Scaler ====================
model = load(MODEL_PATH)
scaler = load(SCALER_PATH)

# ==================== Selected Features ====================
selected_features = [
    'SMOKING',
    'SMOKING_FAMILY_HISTORY',
    'THROAT_DISCOMFORT',
    'BREATHING_ISSUE',
    'STRESS_IMMUNE',
    'ENERGY_LEVEL',
    'IMMUNE_WEAKNESS',
    'FAMILY_HISTORY'
]

# ==================== Routes ====================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/patient_history')
def patient_history():
    return render_template('patient_history.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = []
        for feature in selected_features:
            if feature == 'ENERGY_LEVEL':
                val = float(request.form[feature])
                inputs.append(val)
            else:
                val = request.form[feature]
                inputs.append(1 if val.lower() == 'yes' else 0)

        user_input = np.array(inputs).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input)
        prediction = int(model.predict(user_input_scaled)[0])

        with mlflow.start_run(run_name="Patient_History_Prediction"):
            mlflow.log_params(dict(zip(selected_features, inputs)))
            mlflow.log_metric("prediction", prediction)

        if prediction == 1:
            result = "⚠️ High Risk Detected"
            color = 'red'
        else:
            result = "✅ Low Risk Detected"
            color = 'green'

        return render_template('patient_history.html', result=result, color=color)

    except Exception as e:
        return render_template('patient_history.html', result=f"Error: {e}", color="orange")

# ==================== X-Ray Model ====================
app.config['UPLOAD_FOLDER'] = 'static/uploads'

classes = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
           'Emphysema','Fibrosis','Hernia','Infiltration','Mass','No Finding',
           'Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']
num_classes = len(classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = models.vgg16(weights=None)
vgg.classifier[6] = nn.Linear(4096, num_classes)
vgg.load_state_dict(torch.load("models/best_vgg16_chestxray.pth", map_location=device))
vgg.to(device)
vgg.eval()

predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(img_path, model, classes, device):
    image = Image.open(img_path).convert("RGB")
    image = predict_transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    # all predictions
    preds = {cls: float(prob) for cls, prob in zip(classes, probs)}

    # sort by probability descending and take top 5
    top_preds = dict(sorted(preds.items(), key=lambda item: item[1], reverse=True)[:5])
    return top_preds

# ==================== X-Ray Routes ====================
@app.route('/xray_predict', methods=['GET', 'POST'])
def xray_predict():
    if request.method == 'GET':
        return render_template('xray_upload.html')

    file = request.files['xray_image']
    if not file or file.filename == '':
        return render_template('xray_upload.html', result="No file uploaded.")

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    preds = predict_image(img_path, vgg, classes, device)
    results = [{"label": label, "prob": f"{prob*100:.2f}%"} for label, prob in preds.items()]

    return render_template('xray_upload.html', results=results, image_path=f"static/uploads/{file.filename}")

if __name__ == "__main__":
    app.run(debug=True,port=3000)
