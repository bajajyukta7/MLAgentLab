import streamlit as st
import os
import dotenv
import uuid
import asyncio
import base64
import glob
from PIL import Image
from tool_wrapper import ToolWrapper
from swarm_team import chat_with_agent

dotenv.load_dotenv()

st.set_page_config(
    page_title="MLE Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.markdown("<h2 style='text-align: center;'>🤖 <i>MLE Agent</i></h2>", unsafe_allow_html=True)

# Session state initialization
for key in [
    "session_id", "messages", "last_prompt", "last_response", "last_powershell_command",
    "show_chat", "do_print", "uploaded_image"
]:
    if key not in st.session_state:
        if key == "session_id":
            st.session_state[key] = str(uuid.uuid4())
        elif key == "messages":
            st.session_state[key] = []
        else:
            st.session_state[key] = None if key == "uploaded_image" else False

# --- Sidebar ---
with st.sidebar:
    # Model training inputs
    st.subheader("🧠 Train a Model")

    task_desc = st.text_input("Enter the task description", "train model")
    github_url = st.text_input("Enter GitHub dataset link", "")
    options = st.multiselect("Select the data options", ["Training", "Testing", "Validation"])
    train_split = st.slider("Training data %", 10, 90, 70, 5)
    test_val_split = 100 - train_split
    st.text(f"Testing + Validation: {test_val_split}%")
    model = st.selectbox("Choose a model", ["CNN", "RNN", "SVM", "Random Forest", "Auto Model"], index=4)

    if st.button("Run"):
        training_data = (
            f"Task description: {task_desc}\n"
            f"GitHub Dataset URL: {github_url}\n"
            f"Selected data options: {', '.join(options)}\n"
            f"Training percentage: {train_split}%\n"
            f"Testing + Validation percentage: {test_val_split}%\n"
            f"Model chosen: {model}\n"
        )

        st.session_state.messages.append({"role": "user", "content": training_data})
        with st.chat_message("assistant"):
            try:
                with st.spinner("⚙️ Training the model..."):
                    response = asyncio.run(chat_with_agent(st.session_state.messages))
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.success("Model training has started on CPU due to GPU limitations!")

                st.markdown(ToolWrapper.model_training(training_data))

                model_files = glob.glob("*.keras") + glob.glob("*.h5") + glob.glob("*.pb") + \
                              glob.glob("*.pt") + glob.glob("*.pkl") + glob.glob("*.sav")
                if model_files:
                    model_path = model_files[0]
                    st.success(f"Trained Model: {model_path}")
                    with open(model_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Trained Model",
                            data=f,
                            file_name=os.path.basename(model_path),
                            mime="application/octet-stream"
                        )
                else:
                    st.warning("No model file found after training.")
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")

    st.divider()

    # Upload an image section
    st.subheader("🖼️ Test Your Model Now")
    uploaded_image = st.file_uploader("Upload an image to predict", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        st.session_state.uploaded_image = uploaded_image

    if st.session_state.uploaded_image:
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", width=600)

        image_bytes = st.session_state.uploaded_image.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        st.session_state.messages.append({"role": "user", "content": "Uploaded an image for prediction."})

        with st.chat_message("assistant"):
            try:
                with st.spinner("🔍 Predicting label..."):
                    prediction = ToolWrapper.model_test(encoded_image)
                st.session_state.messages.append({"role": "assistant", "content": prediction})
                st.success(f"🔍 Predicted Label: **{prediction}**")
            except Exception as e:
                st.error(f"⚠️ Error during prediction: {str(e)}")
    else:
        st.write("No image uploaded yet.")

    # Button to clear chat and reset the image
    if st.button("Clear Chat", type="primary"):
        st.session_state.messages = []
        st.session_state.uploaded_image = None
        st.session_state.last_prompt = None
        st.session_state.last_response = None
        st.session_state.last_powershell_command = None

# --- Main Chat Section ---
st.subheader("💬 Agent Chat Log")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.last_prompt = prompt

    with st.chat_message("assistant"):
        try:
            response = asyncio.run(chat_with_agent(st.session_state.messages))
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.last_response = response
            st.markdown(response)
        except Exception as e:
            error_message = f"⚠️ Error generating response: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.session_state.last_response = None
            st.error(error_message)

# --- Regenerate Response Button ---
if st.session_state.get("last_prompt"):
    st.markdown("---")
    if st.button("🔄 Regenerate Response"):
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages.pop()

        st.session_state.messages.append({"role": "user", "content": st.session_state.last_prompt})

        with st.chat_message("user"):
            st.markdown(st.session_state.last_prompt)

        with st.chat_message("assistant"):
            try:
                new_response = asyncio.run(chat_with_agent(st.session_state.messages))
                st.session_state.messages.append({"role": "assistant", "content": new_response})
                st.session_state.last_response = new_response
                st.markdown(new_response)

            except Exception as e:
                error_message = f"⚠️ Error regenerating response: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.session_state.last_response = None
                st.error(error_message)