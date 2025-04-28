import streamlit as st
import os
import uuid
import time
import base64
import glob
from PIL import Image
from tool_wrapper import ToolWrapper

# --- Function to simulate delay ---
def wait_for(seconds: int):
    for i in range(seconds):
        time.sleep(1)
        # st.progress((i + 1) / seconds)

# --- Function to simulate error generation ---
def generate_code_with_error(code_type):
    if code_type == "X":
        return "X.py code with compilation errors"
    elif code_type == "Y":
        return "Y.py code with compilation errors"
    else:
        return "Z.py code - compilable code"

# --- Function to simulate model training ---
def train_model():
    time.sleep(2)
    return "CNN Algorithm Training Started..."

# --- Streamlit Layout ---
st.set_page_config(page_title="MLE Agent", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h2 style='text-align: center;'>🤖 <i>MLE Agent</i></h2>", unsafe_allow_html=True)

# Session state initialization
for key in ["session_id", "messages", "last_prompt", "last_response", "last_powershell_command", "show_chat", "do_print", "uploaded_image"]:
    if key not in st.session_state:
        if key == "session_id":
            st.session_state[key] = str(uuid.uuid4())
        elif key == "messages":
            st.session_state[key] = []
        else:
            st.session_state[key] = None if key == "uploaded_image" else False

# --- Left Side Layout ---
col1, col2 = st.columns([3, 1])

with col1:
    # --- Data Processing View ---
    st.header("🔄 Data Processing")
    with st.expander("Start Data Processing:", expanded=False):
        st.subheader("Processing image data...")

        wait_for(10)  # Simulate processing delay

        st.success("Data processing complete!")
        st.text("Image resized to 256x256 for training.")
        st.text("Data ready for model training!")

    st.divider()

    # --- Model Training View ---
    st.header("🚀 Train Your Model")
    with st.expander("Start training:", expanded=False):
        task_desc = st.text_input("Enter the task description", "train model")
        github_url = st.text_input("Enter GitHub dataset link", "")
        options = st.multiselect("Select the data options", ["Training", "Testing", "Validation"])
        train_split = st.slider("Training data %", 10, 90, 70, 5)
        test_val_split = 100 - train_split
        st.text(f"Testing + Validation: {test_val_split}%")
        model = st.selectbox("Choose a model", ["CNN", "RNN", "SVM", "Random Forest", "Auto Model"], index=4)

        if st.button("Run Training"):
            # Simulate agent processing (resizing image)
            with st.spinner("🔄 Calling Agent for Data Processing..."):
                wait_for(10)
            st.text("Image resized to (256, 256).")

            st.subheader("Training the model...")
            training_data = (
                f"Task description: {task_desc}\n"
                f"GitHub Dataset URL: {github_url}\n"
                f"Selected data options: {', '.join(options)}\n"
                f"Training percentage: {train_split}%\n"
                f"Testing + Validation percentage: {test_val_split}%\n"
                f"Model chosen: {model}\n"
            )

            st.session_state.messages.append({"role": "user", "content": training_data})
            st.text("Training started...")

            # Simulate errors in code generation
            st.text("Generating train_model.py code")
            wait_for(10)  # Simulate wait
            st.text("Compilation errors found. Retrying...")

            st.text("Generating train_model.py code retry attemp 1...")
            wait_for(10)
            st.text("Compilation errors found again. Retrying...")

            st.text("Generating train_model.py code retry attemp 2 - Compilable code...")
            wait_for(10)
            st.text("Code is now compilable!")

            # Simulate successful model training
            with st.spinner("⚙️ Training the model..."):
                wait_for(10)
                st.success("Model training completed successfully!")

            st.markdown("### Trained Model Available for Download")
            model_file = "cat_dog_classifier.h5"
            # with open(model_file, "wb") as f:
            #     f.write(b"dummy_model_data")
            st.download_button(label="⬇️ Download Trained Model", data=open(model_file, "rb"), file_name=model_file, mime="application/octet-stream")

    st.divider()

    # --- Model Testing View ---
    st.header("🧪 Test Your Model")
    with st.expander("Test the model:", expanded=False):
        uploaded_image = st.file_uploader("Upload an image to test", type=["png", "jpg", "jpeg"])

        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", width=200)

            st.text("Processing image for prediction...")
            wait_for(10)  # Simulate wait for processing

            st.text("Predicting label: 'Cat' or 'Dog'...")
            # wait_for(10)
            st.success("Prediction: **Cat**")

# --- Right Sidebar: Real-Time Logs ---
with st.sidebar:
    # --- Real-time Log View ---
    st.subheader("💬 Logs")
    with st.expander("Training Log", expanded=True):
        st.text("Starting Data Processing...")
        st.text("Data resized to (256, 256)...")
        st.text("Starting model training...")
        st.text("Using CNN model for training...")
        st.text("Training completed with an accuracy of 80%")
    
    with st.expander("Test Log", expanded=False):
        st.text("Test image uploaded...")
        st.text("Predicting image label...")
        st.text("Prediction: Cat")

