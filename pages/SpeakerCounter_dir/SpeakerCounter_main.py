import streamlit as st
import time
from SpeakerCounterInference.SpeakerCounter import SpeakerCounter

st.title("Speaker Counter Inference")

model_paths = {
    "ECAPA-TDNN": "SpeakerCounterInference/ecapa_tdnn",
    "XVector": "SpeakerCounterInference/xvector"
}

selected_model = st.selectbox("Select Model", list(model_paths.keys()))

model_path = model_paths[selected_model]

save_dir = "./SpeakerCounterInference/sample_inference_run/"

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])


def stream_data(text_content):
    for line in text_content.split("\n"):
        for word in line.split(" "):
            yield word + " "
            time.sleep(0.02)
        yield "\n"


if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    if st.button("Run Inference"):
        with st.spinner('Running inference...'):
            audio_classifier = SpeakerCounter.from_hparams(source=model_path, savedir=save_dir)
            audio_classifier.classify_file(uploaded_file)
            text_file_path = "SpeakerCounterInference/sample_segment_predictions.txt"

            with open(text_file_path, 'r') as file:
                text_content = file.read()

            st.write("**Inference Results:**")

            text_area = st.empty()

            streamed_text = ""
            for chunk in stream_data(text_content):
                streamed_text += chunk
                text_area.text(streamed_text)
else:
    st.text("Please upload an audio file!")
