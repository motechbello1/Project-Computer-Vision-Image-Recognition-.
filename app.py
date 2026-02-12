import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load Model
# We use caching so it doesn't reload every time you click a button
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cifar10_model.h5')

model = load_model()

# Class names (Must match training order)
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

st.title("👁️ Object Recognition AI")
st.write("Upload an image, and the AI will identify it (from 10 specific categories).")

# 2. File Uploader
file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if file is not None:
    # Display the user's image
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 3. Preprocessing (Crucial Step!)
    # The model expects a 32x32 pixel image with 3 color channels (RGB)
    
    # Resize
    img = image.resize((32, 32))
    
    # Convert to array
    img_array = np.array(img)
    
    # Normalize (0-255 -> 0-1)
    img_array = img_array / 255.0
    
    # Add batch dimension (The model expects a list of images, not just one)
    # Shape becomes (1, 32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. Predict
    if st.button("Identify Object"):
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0]) # Convert logits to probabilities
        
        # Get highest probability class
        class_index = np.argmax(score)
        confidence = 100 * np.max(score)
        
        st.divider()
        st.subheader(f"Prediction: {class_names[class_index]}")
        st.write(f"Confidence: {confidence:.2f}%")
        
        # Show chart of all probabilities
        st.bar_chart(dict(zip(class_names, score.numpy())))