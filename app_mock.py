import streamlit as st
import os
import uuid
import time
import base64
import glob
import random
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
            
            # Check for actual .h5 model files
            h5_files = glob.glob("*.h5")
            
            if h5_files:
                st.success(f"✅ Found {len(h5_files)} trained model(s):")
                
                for model_file in h5_files:
                    file_size = os.path.getsize(model_file) / (1024*1024)  # Size in MB
                    st.info(f"📁 {model_file} ({file_size:.2f} MB)")
                    
                    try:
                        with open(model_file, "rb") as f:
                            st.download_button(
                                label=f"⬇️ Download {model_file}", 
                                data=f.read(), 
                                file_name=model_file, 
                                mime="application/octet-stream",
                                key=f"download_{model_file}"
                            )
                    except Exception as e:
                        st.error(f"Error reading {model_file}: {str(e)}")
                
                # Print detailed info about the models
                st.subheader("📊 Model Details:")
                for model_file in h5_files:
                    with st.expander(f"Details for {model_file}"):
                        file_stats = os.stat(model_file)
                        st.write(f"**File Size:** {file_stats.st_size / (1024*1024):.2f} MB")
                        st.write(f"**Created:** {time.ctime(file_stats.st_ctime)}")
                        st.write(f"**Modified:** {time.ctime(file_stats.st_mtime)}")
                        st.write(f"**Full Path:** {os.path.abspath(model_file)}")
            else:
                st.warning("⚠️ No .h5 model files found in current directory")
                st.info("Models will be saved after training completes")

    st.divider()

    # --- Model Testing View ---
    st.header("🧪 Test Your Model")
    with st.expander("Test the model:", expanded=False):
        uploaded_image = st.file_uploader("Upload an image to test", type=["png", "jpg", "jpeg"])

        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", width=200)

            st.text("Processing image for prediction...")
            wait_for(10)  # Simulate wait for processing

            # Dynamic prediction based on task description and model type
            if task_desc and any(word in task_desc.lower() for word in ["cat", "dog", "animal"]):
                predictions = ["Cat", "Dog"]
                prediction_type = "Animal Classification"
            elif task_desc and "flower" in task_desc.lower():
                predictions = ["Rose", "Tulip", "Daisy", "Sunflower"]
                prediction_type = "Flower Classification"
            elif task_desc and any(word in task_desc.lower() for word in ["fashion", "clothing"]):
                predictions = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat"]
                prediction_type = "Fashion Classification"
            else:
                predictions = ["Class A", "Class B", "Class C"]
                prediction_type = "Generic Classification"
            
            import random
            predicted_class = random.choice(predictions)
            confidence = round(random.uniform(0.75, 0.98), 2)
            
            st.text(f"Predicting using {prediction_type} model...")
            st.success(f"🎯 Prediction: **{predicted_class}** (Confidence: {confidence*100}%)")

# --- Right Sidebar: Real-Time Logs ---
with st.sidebar:
    # --- Real-time Log View ---
    st.subheader("💬 Logs")
    with st.expander("Training Log", expanded=True):
        st.text("Starting Data Processing...")
        st.text("Data resized to (256, 256)...")
        st.text(f"Starting model training with {model}...")
        if model == "CNN":
            st.text("Building convolutional layers...")
            st.text("Adding pooling layers...")
        elif model == "LSTM":
            st.text("Building LSTM layers...")
            st.text("Processing sequential data...")
        elif model == "Random Forest":
            st.text("Training decision trees...")
            st.text("Ensemble learning in progress...")
        
        accuracy = round(random.uniform(0.78, 0.95), 2)
        st.text(f"Training completed with accuracy: {accuracy*100}%")
    
    with st.expander("Test Log", expanded=False):
        st.text("Test image uploaded...")
        st.text("Preprocessing image...")
        st.text("Running inference...")
        if 'predicted_class' in locals():
            st.text(f"Prediction: {predicted_class}")
        else:
            st.text("Prediction: Waiting for test input...")

