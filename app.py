# app.py
"""
SISTEM DEPLOYMENT AGRI-TECH: FLASK FRAMEWORK + HYBRID YOLO & CNN-RF
Backend ini berfungsi sebagai jembatan inferensi.
1. Mendeteksi bagian daun/penyakit menggunakan YOLOv8 (best.pt).
2. Mengekstrak fitur dari bagian terdeteksi menggunakan CNN Keras (model.json + model.weights.h5).
3. Mengklasifikasikan jenis penyakit secara akurat menggunakan Random Forest (random_forest.pkl).
4. Merender hasilnya ke frontend Flask dengan antarmuka premium.
"""

import os
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from ultralytics import YOLO

app = Flask(__name__)

# Konfigurasi Folder Penyimpanan Citra
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Limit ukuran file maksimal 16MB

# Path Model
MODEL_JSON_PATH = "model.json"
MODEL_WEIGHTS_PATH = "model.weights.h5"
YOLO_MODEL_PATH = "best.pt"
RF_MODEL_PATH = "random_forest.pkl"

CLASS_NAMES = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']

# Warna BGR untuk Visualisasi Bounding Box
COLORS = {
    'healthy': (46, 204, 113),      # Hijau (RGB: 46, 204, 113 -> BGR: 113, 204, 46)
    'leaf curl': (52, 152, 219),    # Biru Muda (RGB: 52, 152, 219 -> BGR: 219, 152, 52)
    'leaf spot': (230, 126, 34),    # Oranye (RGB: 230, 126, 34 -> BGR: 34, 126, 230)
    'whitefly': (231, 76, 60),      # Merah (RGB: 231, 76, 60 -> BGR: 60, 76, 231)
    'yellowish': (241, 196, 15)     # Kuning (RGB: 241, 196, 15 -> BGR: 15, 196, 241)
}

# Rekomendasi Penanganan Penyakit/Hama
RECOMMENDATIONS = {
    'healthy': 'Tanaman cabai terlihat sehat. Lanjutkan perawatan rutin seperti penyiraman teratur, pemberian pupuk organik secara berkala, dan pemantauan kondisi daun secara periodik.',
    'leaf curl': 'Daun mengkerut/keriting biasanya dipicu kutu daun (thrips, kutu kebul). Gunakan insektisida nabati (ekstrak daun mimba/sabun kalium) atau pestisida kimia sistemik berbahan aktif abamektin. Kurangi gulma di sekitar area tanaman.',
    'leaf spot': 'Bercak daun disebabkan infeksi jamur Cercospora. Lakukan pemangkasan pada daun bawah yang terinfeksi parah untuk memperbaiki sirkulasi udara, hindari menyiram daun di sore hari, dan semprot fungisida berbahan aktif tembaga oksiklorida.',
    'whitefly': 'Kutu kebul (whiteflies) bertindak sebagai vektor virus Gemini. Pasang perangkap perekat kuning (yellow sticky trap) di sekitar lahan. Jika populasi tinggi, gunakan sabun insektisida atau penyemprotan insektisida kontak.',
    'yellowish': 'Daun menguning bisa karena Virus Gemini (Bule) atau defisiensi hara (Nitrogen). Untuk pencegahan virus, kendalikan hama vektornya (kutu kebul). Lakukan pemupukan tambahan Nitrogen (pupuk urea/urea cair) dan singkirkan tanaman yang sakit parah.'
}

# --- 1. Memuat Model CNN Keras & Feature Extractor ---
keras_model = None
feature_extractor = None
if os.path.exists(MODEL_JSON_PATH) and os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"[INFO] Memuat arsitektur CNN dari {MODEL_JSON_PATH}...")
    try:
        with open(MODEL_JSON_PATH, "r") as json_file:
            model_json = json_file.read()
        keras_model = model_from_json(model_json)
        keras_model.load_weights(MODEL_WEIGHTS_PATH)
        
        # Inisialisasi model graph dengan dummy input
        keras_model(tf.zeros((1, 100, 100, 3)))
        
        # Temukan layer Dense 128-unit secara dinamis untuk ekstraksi fitur
        dense_layer = None
        for layer in reversed(keras_model.layers):
            if isinstance(layer, tf.keras.layers.Dense) and layer.units == 128:
                dense_layer = layer
                break
        
        if dense_layer is None:
            # Fallback pencarian berdasarkan nama
            for layer in keras_model.layers:
                if "dense" in layer.name and getattr(layer, 'units', None) == 128:
                    dense_layer = layer
                    break
        
        if dense_layer is not None:
            # Bangun model ekstraktor fitur Functional
            inputs = tf.keras.Input(shape=(100, 100, 3))
            x = inputs
            for layer in keras_model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                x = layer(x)
                if layer.name == dense_layer.name:
                    break
            feature_extractor = tf.keras.Model(inputs=inputs, outputs=x)
            print(f"[INFO] Sukses membangun Feature Extractor CNN berbasis layer '{dense_layer.name}'")
        else:
            print("[ERROR] Layer dense 128 unit tidak ditemukan pada model CNN Keras!")
    except Exception as e:
        print(f"[ERROR] Gagal memuat model Keras: {e}")
else:
    print("[WARNING] File model CNN Keras tidak ditemukan!")

# --- 2. Memuat Model YOLOv8 ---
yolo_model = None
if os.path.exists(YOLO_MODEL_PATH):
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"[INFO] Sukses memuat model YOLO dari {YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Gagal memuat model YOLO: {e}")
else:
    print("[WARNING] Model YOLO 'best.pt' tidak ditemukan!")

# --- 3. Memuat Model Random Forest ---
rf_model = None
if os.path.exists(RF_MODEL_PATH):
    try:
        rf_model = joblib.load(RF_MODEL_PATH)
        print(f"[INFO] Sukses memuat model Random Forest dari {RF_MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Gagal memuat model Random Forest: {e}")
else:
    print("[WARNING] Model Random Forest 'random_forest.pkl' tidak ditemukan!")

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
            
            # Load citra untuk diproses
            img = cv2.imread(input_image_path)
            h, w, _ = img.shape
            output_img = img.copy()
            
            detections = []
            detection_done = False
            
            # Jalankan deteksi YOLO jika model tersedia
            if yolo_model is not None and feature_extractor is not None and rf_model is not None:
                try:
                    results = yolo_model(input_image_path)
                    boxes = results[0].boxes
                    
                    # 1. Jika YOLO berhasil mendeteksi objek
                    if len(boxes) > 0:
                        for idx, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Batasi koordinat agar berada di dalam gambar
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            crop = img[y1:y2, x1:x2]
                            if crop.size == 0:
                                continue
                            
                            # Inferensi menggunakan CNN + RF
                            crop_resized = cv2.resize(crop, (100, 100))
                            crop_array = crop_resized.astype('float32') / 255.0
                            crop_array = np.expand_dims(crop_array, axis=0)
                            
                            # Ekstrak fitur CNN & Prediksi RF
                            features = feature_extractor.predict(crop_array, verbose=0)
                            rf_probs = rf_model.predict_proba(features)
                            class_idx = np.argmax(rf_probs[0])
                            
                            label = CLASS_NAMES[class_idx]
                            confidence = rf_probs[0][class_idx] * 100
                            
                            # BGR color dari kamus warna (default merah jika tidak ditemukan)
                            color_rgb = COLORS.get(label, (231, 76, 60))
                            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0]) # Konversi RGB ke BGR untuk OpenCV
                            
                            # Gambar bounding box dan label di citra output
                            cv2.rectangle(output_img, (x1, y1), (x2, y2), color_bgr, 3)
                            
                            # Label text
                            text = f"#{idx+1} {label.title()} ({confidence:.1f}%)"
                            
                            # Skala font dinamis berdasarkan ukuran box
                            box_height = y2 - y1
                            font_scale = 0.5 if box_height < 100 else 0.7
                            thickness = 1 if box_height < 100 else 2
                            
                            # Tulis teks di atas kotak
                            cv2.putText(output_img, text, (x1, max(15, y1 - 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, thickness, cv2.LINE_AA)
                            
                            detections.append({
                                'id': idx + 1,
                                'label': label.title(),
                                'confidence': f"{confidence:.1f}",
                                'color': f"rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})",
                                'recommendation': RECOMMENDATIONS.get(label, '')
                            })
                        detection_done = True
                    
                    # 2. Fallback: Jika YOLO tidak mendeteksi box apa pun, klasifikasi seluruh gambar
                    else:
                        img_resized = cv2.resize(img, (100, 100))
                        img_array = img_resized.astype('float32') / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        features = feature_extractor.predict(img_array, verbose=0)
                        rf_probs = rf_model.predict_proba(features)
                        class_idx = np.argmax(rf_probs[0])
                        
                        label = CLASS_NAMES[class_idx]
                        confidence = rf_probs[0][class_idx] * 100
                        
                        color_rgb = COLORS.get(label, (231, 76, 60))
                        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                        
                        # Gambar banner di atas untuk indikasi global
                        banner_height = int(h * 0.12) if h * 0.12 > 40 else 40
                        cv2.rectangle(output_img, (0, 0), (w, banner_height), color_bgr, -1)
                        
                        text = f"Global: {label.title()} ({confidence:.1f}%)"
                        font_scale = banner_height / 55.0
                        thickness = int(banner_height / 25) if banner_height / 25 > 1 else 1
                        cv2.putText(output_img, text, (15, int(banner_height * 0.7)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                        
                        detections.append({
                            'id': 'Global',
                            'label': label.title(),
                            'confidence': f"{confidence:.1f}",
                            'color': f"rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})",
                            'recommendation': RECOMMENDATIONS.get(label, '') + " (Hasil analisis menyeluruh pada seluruh citra karena tidak ada area bergejala spesifik yang terdeteksi)."
                        })
                        detection_done = True
                        
                except Exception as e:
                    print(f"[ERROR] Masalah saat inferensi model: {e}")
            
            # Jika model gagal dijalankan, fallback ke gambar asli
            if not detection_done:
                output_img = img.copy()
                detections.append({
                    'id': 'Error',
                    'label': 'Sistem Error',
                    'confidence': '0.0',
                    'color': 'rgb(231,76,60)',
                    'recommendation': 'Terjadi kesalahan sistem ketika memproses gambar. Pastikan semua file model terpasang dengan benar.'
                })
            
            # Simpan hasil gambar terproses
            output_filename = "result_" + filename
            output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_image_path, output_img)
            
            return render_template('index.html', 
                                   original_img=input_image_path, 
                                   predicted_img=output_image_path,
                                   detections=detections,
                                   detection_done=True)
            
    return render_template('index.html', detection_done=False)

if __name__ == '__main__':
    # Menjalankan server lokal pada port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)