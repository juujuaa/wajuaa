import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Klasifikasi Gambar: Hiu atau Burung", layout="wide")

with st.sidebar:
    st.header("Pengaturan Model")
    st.write("1. Upload Model (.h5 atau .tflite)")
    uploaded_model = st.file_uploader("Pilih file model...", type=["h5", "tflite"], label_visibility="collapsed")
    st.caption("1GB per file • H5, TFLITE")

st.title("🦈 Klasifikasi Gambar: Hiu atau Burung? 🦅")
st.subheader("Aplikasi Versi Ringan dan 100% Stabil Berhasil Dimuat!")
st.write("") 

st.write("2. Upload Gambar (Hiu/Burung)")
uploaded_image = st.file_uploader("Pilih file gambar...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
st.caption("1GB per file • JPG, PNG")
st.write("") 

if uploaded_model is None:
    st.info("Silahkan upload file model (.h5/.tflite) kamu di sidebar sebelah kiri untuk mengaktifkan fungsi klasifikasi.")
else:
    interpreter = tf.lite.Interpreter(model_content=uploaded_model.read())
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Gambar yang diunggah", width=300)
        
        img_resized = image.resize((150, 150))
        img_array = np.array(img_resized, dtype=np.float32)
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        img_array = img_array / 255.0
        img_input = np.expand_dims(img_array, axis=0)
        
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