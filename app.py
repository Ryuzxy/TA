# app.py
"""
SISTEM DEPLOYMENT AGRI-TECH: FLASK FRAMEWORK
Backend ini berfungsi sebagai jembatan inferensi. Menerima request HTTP berupa 
citra daun/tanaman cabai, memprosesnya melalui Keras CNN model, dan merender hasilnya ke frontend.
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# Konfigurasi Folder Penyimpanan Citra
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Limit ukuran file maksimal 16MB

# Memuat Model Keras yang telah dilatih (Inisialisasi Global untuk menghemat memori)
MODEL_JSON_PATH = "model.json"
MODEL_WEIGHTS_PATH = "model.weights.h5"
CLASS_NAMES = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']

keras_model = None
if os.path.exists(MODEL_JSON_PATH) and os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"[INFO] Sukses memuat model CNN dari {MODEL_JSON_PATH}")
    try:
        with open(MODEL_JSON_PATH, "r") as json_file:
            model_json = json_file.read()
        keras_model = model_from_json(model_json)
        keras_model.load_weights(MODEL_WEIGHTS_PATH)
    except Exception as e:
        print(f"[ERROR] Gagal memuat model Keras: {e}")
else:
    print("[WARNING] Model 'model.json' atau 'model.weights.h5' tidak ditemukan!")

# Ekstensi file yang diizinkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Validasi ketersediaan file dalam request
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Mengamankan nama file dan menyimpan ke direktori statis
            filename = secure_filename(file.filename)
            input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_image_path)
            
            # Default values
            label = "Tidak Diketahui"
            confidence = 0.0
            
            if keras_model is not None:
                # --- PROSES INFERENSI CNN ---
                img = cv2.imread(input_image_path)
                h, w, _ = img.shape
                
                # Preprocess untuk model (ukuran 100x100, normalisasi)
                img_resized = cv2.resize(img, (100, 100))
                img_array = img_resized.astype('float32') / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                preds = keras_model.predict(img_array)
                class_idx = np.argmax(preds[0])
                label = CLASS_NAMES[class_idx]
                confidence = preds[0][class_idx] * 100
                
                # Menggambar teks hasil prediksi pada citra
                output_img = img.copy()
                text = f"{label.title()} ({confidence:.1f}%)"
                
                # Gambar banner berwarna di bagian atas
                banner_height = int(h * 0.12) if h * 0.12 > 40 else 40
                # Hijau untuk sehat, orange/merah untuk penyakit
                banner_color = (46, 204, 113) if label == 'healthy' else (41, 128, 185) # BGR
                cv2.rectangle(output_img, (0, 0), (w, banner_height), banner_color, -1)
                
                # Overlay teks
                font_scale = banner_height / 50.0
                thickness = int(banner_height / 25) if banner_height / 25 > 1 else 1
                cv2.putText(output_img, text, (15, int(banner_height * 0.7)), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            else:
                output_img = cv2.imread(input_image_path)
                
            output_filename = "result_" + filename
            output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_image_path, output_img)
            
            # Mengirimkan path citra asli dan citra hasil prediksi ke frontend HTML
            return render_template('index.html', 
                                   original_img=input_image_path, 
                                   predicted_img=output_image_path,
                                   label=label.title(),
                                   confidence=f"{confidence:.1f}",
                                   detection_done=True)
            
    return render_template('index.html', detection_done=False)

if __name__ == '__main__':
    # Menjalankan server lokal pada port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)