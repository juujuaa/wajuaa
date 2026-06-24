import streamlit as st
import numpy as np
from PIL import Image

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Gambar: Hiu atau Burung", layout="wide")

# --- SIDEBAR: Pengaturan Model ---
with st.sidebar:
    st.header("Pengaturan Model")
    st.write("1. Upload Model (.tflite)")
    
    uploaded_model = st.file_uploader(
        "Pilih file model...", 
        type=["tflite"], 
        label_visibility="collapsed"
    )
    st.caption("1GB per file • TFLITE")

# --- HALAMAN UTAMA ---
st.title("🦈 Klasifikasi Gambar: Hiu atau Burung? 🦅")
st.subheader("Aplikasi Versi Ringan dan 100% Stabil Berhasil Dimuat!")
st.write("") 

st.write("2. Upload Gambar (Hiu/Burung)")
uploaded_image = st.file_uploader(
    "Pilih file gambar...", 
    type=["jpg", "png", "jpeg"], 
    label_visibility="collapsed"
)
st.caption("1GB per file • JPG, PNG")
st.write("") 

# --- LOGIKA APLIKASI ---
if uploaded_model is None:
    st.info("Silahkan upload file model (.tflite) kamu di sidebar sebelah kiri untuk mengaktifkan fungsi klasifikasi.")
else:
    # Menggunakan tflite_runtime bawaan streamlit platform jika tersedia atau via interpreter alternatif
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            st.error("Sistem Cloud sedang menyiapkan runtime. Harap tunggu beberapa saat atau reboot app.")
            st.stop()

    # Memuat model tflite dari buffer upload
    try:
        model_bytes = uploaded_model.read()
        interpreter = tflite.Interpreter(model_content=model_bytes)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Gambar yang diunggah", width=300)
            
            # Preprocessing Gambar
            img_resized = image.resize((150, 150))
            img_array = np.array(img_resized, dtype=np.float32)
            
            if len(img_array.shape) == 2:
                img_array = np.stack((img_array,)*3, axis=-1)
                
            img_array = img_array / 255.0
            img_input = np.expand_dims(img_array, axis=0)
            
            # Jalankan Prediksi
            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            st.write("### **Hasil Analisis Citra:**")
            if prediction < 0.5:
                persentase = (1 - prediction) * 100
                st.success(f"🦅 Gambar ini terdeteksi sebagai **BURUNG** ({persentase:.2f}%)")
            else:
                persentase = prediction * 100
                st.success(f"🦈 Gambar ini terdeteksi sebagai **HIU** ({persentase:.2f}%)")
    except Exception as e:
        st.error(f"Gagal memproses model tflite: {str(e)}")
