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


# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "eeg_cnn_model.h5")

st.set_page_config(
    page_title="EEG Seizure Detection",
    layout="wide"
)

st.title("EEG Seizure Detection with CNN + GPT-4V")


# =========================
# FIX KERAS VERSION COMPATIBILITY
# =========================

class CustomDense(Dense):
    """
    Dùng để load model .h5 được lưu bằng Keras version khác.
    Bỏ qua quantization_config nếu có, không ảnh hưởng weight/model.
    """
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy file model tại:\n{MODEL_PATH}\n\n"
            "Hãy kiểm tra file model/eeg_cnn_model.h5 đã nằm đúng thư mục chưa."
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
    X_scaled = scaler.fit_transform(X)

    X_scaled = X_scaled.reshape(X_scaled.shape[0], 178, 1)

    return X_scaled, y


# =========================
# PREDICT
# =========================

def predict(cnn_model, sample):
    output = cnn_model.predict(sample, verbose=0)

    prob = float(output[0][0])
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

    saliency_min = saliency.min()
    saliency_max = saliency.max()

    saliency = (saliency - saliency_min) / (
        saliency_max - saliency_min + 1e-10
    )

    return saliency


# =========================
# IMPORTANT REGIONS
# =========================

def get_important_regions(saliency, threshold_percentile=85, min_length=2):
    """
    Tìm các vùng CNN chú ý mạnh dựa trên percentile.
    Ví dụ threshold 85 nghĩa là lấy top 15% điểm saliency cao nhất.
    """
    threshold = np.percentile(saliency, threshold_percentile)
    important = saliency >= threshold

    regions = []
    start = None

    for i, is_important in enumerate(important):
        if is_important and start is None:
            start = i

        elif not is_important and start is not None:
            end = i - 1

            if end - start + 1 >= min_length:
                regions.append((start, end))

            start = None

    if start is not None:
        end = len(important) - 1

        if end - start + 1 >= min_length:
            regions.append((start, end))

    return regions, threshold


def build_clinical_summary(pred_text, prob, regions, signal):
    """
    Tạo đoạn giải thích ngắn theo hướng hỗ trợ bác sĩ.
    Không thay thế chẩn đoán y khoa.
    """
    if len(regions) == 0:
        region_text = "Không phát hiện vùng saliency nổi bật rõ ràng."
    else:
        region_list = []
        for start, end in regions:
            min_amp = float(np.min(signal[start:end + 1]))
            max_amp = float(np.max(signal[start:end + 1]))

            region_list.append(
                f"- Time point {start}–{end}: biên độ khoảng {min_amp:.2f} đến {max_amp:.2f}"
            )

        region_text = "\n".join(region_list)

    summary = f"""
### Clinical Support Summary

**Kết quả CNN:** {pred_text}  
**Xác suất dự đoán:** {prob:.4f}

**Các vùng tín hiệu CNN chú ý mạnh:**

{region_text}

**Diễn giải hỗ trợ bác sĩ:**

Mô hình CNN đang đánh giá mẫu EEG này thuộc nhóm **{pred_text}** với xác suất **{prob:.4f}**.  
Các vùng được đánh dấu là những đoạn có giá trị Saliency cao, tức là các điểm tín hiệu có ảnh hưởng lớn hơn đến quyết định của mô hình.

Nếu dự đoán là **Seizure**, bác sĩ nên chú ý các đoạn có:
- Biên độ thay đổi đột ngột.
- Pha giảm sâu hoặc tăng nhanh bất thường.
- Cụm dao động ngắn, sắc, hoặc biến thiên mạnh.
- Vùng được tô nền trên biểu đồ vì đây là nơi CNN tập trung nhiều hơn.

**Lưu ý:** Kết quả này chỉ đóng vai trò hỗ trợ sàng lọc và giải thích mô hình AI. Không dùng thay thế kết luận lâm sàng của bác sĩ.
"""

    return summary


# =========================
# DRAW EEG IMAGE
# =========================

def draw_eeg(signal, saliency, pred_text, prob, regions):
    fig, ax = plt.subplots(figsize=(13, 4.5))

    ax.plot(
        signal,
        color="black",
        linewidth=1.5,
        label="EEG Signal"
    )

    # Tô nền các vùng quan trọng
    for start, end in regions:
        ax.axvspan(
            start,
            end,
            alpha=0.18
        )

    # Saliency scatter
    points = ax.scatter(
        np.arange(len(signal)),
        signal,
        c=saliency,
        cmap="jet",
        s=45,
        vmin=0,
        vmax=1,
        label="CNN Saliency"
    )

    ax.set_title(
        f"Prediction: {pred_text} | Probability: {prob:.4f}",
        fontsize=14
    )

    ax.set_xlabel("Time Point")
    ax.set_ylabel("Amplitude")

    ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(points, ax=ax)
    cbar.set_label("CNN Attention / Saliency Score")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    ax.legend(loc="upper right")

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


def ask_gpt4v(image_path, pred_text, prob, regions):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return (
            "Chưa có OPENAI_API_KEY.\n\n"
            "Bạn cần set API key trước khi dùng GPT-4V."
        )

    client = OpenAI(api_key=api_key)

    image64 = image_to_base64(image_path)

    region_text = ", ".join([f"{s}-{e}" for s, e in regions])
    if not region_text:
        region_text = "No clear highlighted region"

    prompt = f"""
You are an AI assistant supporting EEG interpretation for clinicians.

Analyze this EEG waveform image with CNN saliency overlay.

Important context:
- This is NOT a medical diagnosis.
- The CNN prediction is: {pred_text}
- CNN probability is: {prob:.4f}
- Highlighted important time regions are: {region_text}
- Black line is EEG waveform.
- Colored points represent CNN saliency score from 0 to 1.
- Higher saliency means the point influenced the CNN prediction more strongly.

Please explain in a clinician-support style.

Output format:
1. Overall AI Result
2. Important EEG Regions
3. Waveform Characteristics
4. CNN Saliency Interpretation
5. Clinical Support Summary
6. Safety Note
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

        threshold_percentile = st.slider(
            "Ngưỡng hiển thị vùng CNN chú ý mạnh",
            min_value=70,
            max_value=95,
            value=85,
            step=5,
            help="Giá trị càng cao thì chỉ hiển thị các điểm saliency nổi bật nhất."
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

            regions, saliency_threshold = get_important_regions(
                saliency,
                threshold_percentile=threshold_percentile,
                min_length=2
            )

            st.write(
                f"Saliency threshold top region: percentile {threshold_percentile} "
                f"= {saliency_threshold:.4f}"
            )

            image_path = draw_eeg(
                signal,
                saliency,
                pred_text,
                prob,
                regions
            )

            st.subheader("EEG + CNN Saliency Map")
            st.image(image_path, use_container_width=True)

            st.subheader("Clinical Support Explanation")
            clinical_summary = build_clinical_summary(
                pred_text,
                prob,
                regions,
                signal
            )
            st.markdown(clinical_summary)

            if use_gpt:
                st.subheader("GPT-4V Explanation")

                with st.spinner("GPT-4V đang phân tích..."):
                    result = ask_gpt4v(
                        image_path,
                        pred_text,
                        prob,
                        regions
                    )

                st.write(result)

    except Exception as e:
        st.error("Có lỗi khi xử lý file EEG.")
        st.code(str(e))
