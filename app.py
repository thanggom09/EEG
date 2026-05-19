import os
import base64
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from openai import OpenAI


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "eeg_cnn_model.h5")


st.set_page_config(
    page_title="EEG Seizure Detection",
    layout="wide"
)

st.title("EEG Seizure Detection with CNN + GPT-4V")


class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)


@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy file model tại:\n{MODEL_PATH}"
        )

    cnn_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False,
        custom_objects={
            "Dense": CustomDense
        }
    )

    return cnn_model


try:
    st.write("Đường dẫn model đang dùng:")
    st.code(MODEL_PATH)

    cnn_model = load_cnn_model()
    st.success("Đã load model CNN thành công.")

except Exception as e:
    st.error("Không load được model.")
    st.code(str(e))
    st.stop()


# =========================
# PREPROCESS DATA
# =========================

def preprocess(df):
    # Xóa cột Unnamed nếu có
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if "y" not in df.columns:
        raise ValueError("File CSV phải có cột y.")

    y = df["y"].apply(lambda v: 1 if v == 1 else 0).values
    X = df.drop(columns=["y"]).values

    if X.shape[1] != 178:
        raise ValueError(
            f"Model cần 178 cột EEG, nhưng file hiện có {X.shape[1]} cột."
        )

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Shape cho CNN: samples, time_steps, channels
    X = X.reshape(X.shape[0], 178, 1)

    return X, y


# =========================
# PREDICT
# =========================

def predict(cnn_model, sample):
    prob = cnn_model.predict(sample, verbose=0)[0][0]
    label = 1 if prob >= 0.5 else 0

    return label, prob


# =========================
# SALIENCY MAP
# =========================

def make_saliency(cnn_model, sample):
    x = tf.convert_to_tensor(sample, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x)
        output = cnn_model(x)
        loss = output[:, 0]

    grads = tape.gradient(loss, x)

    if grads is None:
        raise ValueError("Không tạo được gradient cho Saliency Map.")

    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()[0]

    saliency = (saliency - saliency.min()) / (
        saliency.max() - saliency.min() + 1e-10
    )

    return saliency


# =========================
# DRAW EEG IMAGE
# =========================

def draw_eeg(signal, saliency, pred_text, prob):
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        signal,
        color="black",
        linewidth=1.5
    )

    points = ax.scatter(
        np.arange(len(signal)),
        signal,
        c=saliency,
        cmap="jet",
        s=45
    )

    ax.set_title(
        f"Prediction: {pred_text} | Probability: {prob:.4f}"
    )

    ax.set_xlabel("Time Point")
    ax.set_ylabel("Amplitude")

    fig.colorbar(points, ax=ax, label="Saliency Importance")
    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=300)
    plt.close(fig)

    return tmp.name


# =========================
# GPT-4V / VISION
# =========================

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_gpt4v(image_path):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return (
            "Chưa có OPENAI_API_KEY.\n\n"
            "Bạn cần set API key trước khi dùng GPT-4V."
        )

    client = OpenAI(api_key=api_key)

    image64 = image_to_base64(image_path)

    prompt = """
You are an AI assistant specialized in EEG waveform interpretation.

Analyze this EEG signal with saliency map overlay.

Context:
- Black line is EEG waveform.
- Colored points are saliency values from CNN.
- High-saliency points strongly influence CNN seizure prediction.
- Do not provide medical diagnosis.

Output:
- Important Regions:
- Waveform Characteristics:
- CNN-based Explanation:
- Clinical Support Summary:
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image64}"
                    }
                ]
            }
        ]
    )

    return response.output_text


# =========================
# STREAMLIT UI
# =========================

uploaded_file = st.file_uploader(
    "Upload file EEG CSV",
    type=["csv"]
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        X, y = preprocess(df)

        st.write("Số mẫu EEG:", len(df))
        st.dataframe(df.head())

        index = st.number_input(
            "Chọn dòng EEG",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1
        )

        use_gpt = st.checkbox("Gọi GPT-4V để giải thích ảnh")

        if st.button("Analyze"):
            sample = X[index:index + 1]
            signal = sample.reshape(-1)

            pred, prob = predict(cnn_model, sample)

            if pred == 1:
                pred_text = "Seizure"
            else:
                pred_text = "Non-seizure"

            st.subheader("CNN Prediction")

            c1, c2, c3 = st.columns(3)

            c1.metric("True Label", int(y[index]))
            c2.metric("Prediction", pred_text)
            c3.metric("Seizure Probability", f"{prob:.4f}")

            saliency = make_saliency(cnn_model, sample)

            image_path = draw_eeg(
                signal,
                saliency,
                pred_text,
                prob
            )

            st.subheader("EEG + Saliency Map")
            st.image(image_path, use_container_width=True)

            if use_gpt:
                st.subheader("GPT-4V Explanation")

                with st.spinner("GPT-4V đang phân tích..."):
                    result = ask_gpt4v(image_path)

                st.write(result)

    except Exception as e:
        st.error("Có lỗi khi xử lý file EEG.")
        st.code(str(e))
