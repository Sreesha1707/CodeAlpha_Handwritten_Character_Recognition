
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Handwritten Recognition AI",
    page_icon="🔤",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
}

.main .block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(120deg, #312e81, #4f46e5, #8b5cf6);
    padding: 35px;
    border-radius: 22px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(79,70,229,0.2);
}

.hero h1 {
    color: white;
    font-size: 36px;
}

.hero p {
    color: #e0e7ff;
    font-size: 16px;
}

.info-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e0e7ff;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.04);
    min-height: 115px;
}

.section-title {
    color: #3730a3;
    font-size: 25px;
    font-weight: bold;
    margin-top: 30px;
    margin-bottom: 8px;
}

.workspace {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e0e7ff;
    min-height: 260px;
}

.result-card {
    background: linear-gradient(135deg, #eef2ff, #ede9fe);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #c7d2fe;
}

.result-character {
    font-size: 75px;
    font-weight: 800;
    color: #4338ca;
}

div.stButton > button {
    background: #4338ca;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "handwritten_character_model.keras"
    )

model = load_model()

labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🔤 Handwritten Character Recognition</h1>
    <p>Transform handwritten digits and letters into digital predictions using AI</p>
    <p>✨ Supports digits 0–9 and uppercase letters A–Z</p>
</div>
""", unsafe_allow_html=True)


# ---------------- INFORMATION CARDS ----------------

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <h2>🔢 10 Digits</h2>
        <p>Recognizes numbers from 0 to 9</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h2>🔠 26 Letters</h2>
        <p>Recognizes uppercase letters A to Z</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <h2>🤖 AI Powered</h2>
        <p>Uses a trained CNN deep learning model</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- RECOGNITION WORKSPACE ----------------

st.markdown(
    '<div class="section-title">Recognition Workspace</div>',
    unsafe_allow_html=True
)

st.write("Select a recognition task and upload your handwritten image.")

task = st.radio(
    "Recognition Task",
    ["Digit Recognition", "Character Recognition"],
    horizontal=True
)

if task == "Digit Recognition":
    st.caption("Upload a single handwritten digit from 0 to 9.")
    allowed_indices = list(range(0, 10))
else:
    st.caption("Upload a single handwritten uppercase letter from A to Z.")
    allowed_indices = list(range(10, 36))


# ---------------- INPUT AND RESULT ----------------

left, right = st.columns(2, gap="large")

with left:
    st.markdown("""
    <div class="workspace">
        <h3>01 · INPUT HANDWRITING</h3>
        <p>Upload your handwritten image below.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        key="handwriting_upload"
    )

with right:
    st.markdown("""
    <div class="workspace">
        <h3>02 · RECOGNITION RESULT</h3>
        <p>Your predicted character will appear here.</p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is None:
        st.info("Your prediction will appear here after uploading an image.")


# ---------------- PREDICTION ----------------

if uploaded_file is not None:

    original = Image.open(uploaded_file).convert("L")
    original = ImageOps.autocontrast(original)

    inverted = ImageOps.invert(original)

    binary = inverted.point(
        lambda p: 255 if p > 80 else 0
    )

    bbox = binary.getbbox()

    if bbox is None:
        st.error("No character detected. Please upload another image.")

    else:
        digit = binary.crop(bbox)
        digit.thumbnail((20, 20))

        canvas = Image.new("L", (28, 28), 0)

        x = (28 - digit.width) // 2
        y = (28 - digit.height) // 2

        canvas.paste(digit, (x, y))

        img_array = np.array(
            canvas, dtype=np.float32
        ) / 255.0

        img_array = img_array.reshape(
            1, 28, 28, 1
        )

        prediction = model.predict(
            img_array, verbose=0
        )[0]

        # Select only the classes for the chosen task
        selected_predictions = prediction[allowed_indices]

        selected_index = int(
            np.argmax(selected_predictions)
        )

        predicted_index = allowed_indices[selected_index]

        predicted_character = labels[predicted_index]

        # Confidence within the selected task
        confidence = float(
            selected_predictions[selected_index] /
            np.sum(selected_predictions) * 100
        )

        st.markdown("---")

        image_col, result_col = st.columns(2, gap="large")

        with image_col:
            st.subheader("🖼️ Uploaded Handwriting")
            st.image(
                original,
                caption="Original uploaded image",
                use_container_width=True
            )

        with result_col:
            st.subheader("🎯 Prediction Output")

            st.markdown(f"""
            <div class="result-card">
                <p style="font-size:18px;color:#475569;">
                    Recognized { 'Digit' if task == 'Digit Recognition' else 'Character' }
                </p>
                <div class="result-character">
                    {predicted_character}
                </div>
                <p style="font-size:21px;color:#4338ca;font-weight:bold;">
                    Confidence: {confidence:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("View processed 28 × 28 image"):
            st.image(
                canvas,
                caption="Image processed for the CNN model"
            )


# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown("""
<p style="text-align:center;color:#64748b;">
    Handwritten Character Recognition |
    Powered by TensorFlow and CNN
</p>
""", unsafe_allow_html=True)
