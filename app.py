import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    "CRICKET-4-(240 X 310)-zeyam-100.00.keras"
)

classes = [
    "cover Drive",
    "Leg Glance / Flick",
    "Pull Shot",
    "Sweep"
]

# The trained model expects images in (height, width, channels) format.
# PIL's resize takes (width, height), so for 240x310 images we must pass (310, 240).
IMG_SIZE = (310, 240)


st.title("🏏 Cricket Image Classification")
st.write("Upload or take a picture and let the model predict the cricket class.")

# Take picture from camera
camera_image = st.camera_input("Take a picture")

# Or upload an image
uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

image = None

if camera_image is not None:
    image = Image.open(camera_image)

elif uploaded_image is not None:
    image = Image.open(uploaded_image)


if image is not None:

    # Display image
    st.image(
        image,
        caption="Input Image",
        use_container_width=True
    )

    # Resize image
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)

    # Convert to numpy array
    img_array = np.array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)

    # Get predicted class
    predicted_index = np.argmax(predictions[0])

    predicted_class = classes[predicted_index]

    # Get confidence
    confidence = predictions[0][predicted_index] * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
