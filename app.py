import streamlit as st
import os
import dotenv
import uuid
import asyncio
import base64
import glob
import time
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

# --- Main Content (Left side) ---
col1, col2 = st.columns([3, 1])

with col1:
    # --- Model training inputs ---
    st.header("Train Your Model")

    with st.expander("Start training:", expanded=False):
        task_desc = st.text_input("Enter the task description", "train model")
        github_url = st.text_input("Enter GitHub dataset link", "")
        options = st.multiselect("Select the data options", ["Training", "Testing", "Validation"])
        train_split = st.slider("Training data %", 10, 90, 70, 5)
        test_val_split = 100 - train_split
        st.text(f"Testing + Validation: {test_val_split}%")
        model = st.selectbox("Choose a model", [
            "CNN", "RNN", "LSTM", "GRU", "Transformer", "BERT", "GPT",
            "SVM", "Random Forest", "Decision Tree", "Gradient Boosting", "XGBoost",
            "AdaBoost", "Naive Bayes", "K-Means", "KNN", "Linear Regression",
            "Logistic Regression", "Ridge Regression", "Lasso Regression",
            "ElasticNet", "Neural Network", "AutoEncoder", "GAN", "ResNet",
            "Auto Model"
        ], index=25)  # Auto Model is now at index 25

        if st.button("Run"):
            training_data = (
                f"Task description: {task_desc}\n"
                f"GitHub Dataset URL: {github_url}\n"
                f"Selected data options: {', '.join(options)}\n"
                f"Training percentage: {train_split}%\n"
                f"Testing + Validation percentage: {test_val_split}%\n"
                f"Model chosen: {model}\n"
            )

            # If Auto Model is selected, call agent to decide the best model
            if model == "Auto Model":
                st.info("🤖 Auto Model selected - asking agent to recommend the best model...")
                auto_model_prompt = (
                    f"You are a machine learning expert. Based on this task: '{task_desc}' and data options: {', '.join(options)}, "
                    f"you must choose ONE of these machine learning models: "
                    f"CNN, RNN, LSTM, GRU, Transformer, BERT, GPT, SVM, Random Forest, Decision Tree, "
                    f"Gradient Boosting, XGBoost, AdaBoost, Naive Bayes, K-Means, KNN, Linear Regression, "
                    f"Logistic Regression, Ridge Regression, Lasso Regression, ElasticNet, Neural Network, "
                    f"AutoEncoder, GAN, ResNet. "
                    f"Do NOT suggest any other models or agent names. "
                    f"RESPOND WITH ONLY THE MODEL NAME from the above list. "
                    f"Guidelines: For image tasks use CNN/ResNet/GAN. For text use RNN/LSTM/GRU/Transformer/BERT/GPT. "
                    f"For tabular data use Random Forest/XGBoost/SVM. For clustering use K-Means. "
                    f"For regression use Linear/Ridge/Lasso Regression."
                )
                
                auto_model_messages = [{"role": "user", "content": auto_model_prompt}]
                
                with st.spinner("🧠 Agent is analyzing and selecting the best model..."):
                    auto_model_response = asyncio.run(chat_with_agent(auto_model_messages))
                
                # Extract just the model name and validate it's one of our allowed models
                allowed_models = [
                    "CNN", "RNN", "LSTM", "GRU", "Transformer", "BERT", "GPT",
                    "SVM", "Random Forest", "Decision Tree", "Gradient Boosting", "XGBoost",
                    "AdaBoost", "Naive Bayes", "K-Means", "KNN", "Linear Regression",
                    "Logistic Regression", "Ridge Regression", "Lasso Regression",
                    "ElasticNet", "Neural Network", "AutoEncoder", "GAN", "ResNet"
                ]
                model_name = auto_model_response.split('\n')[0].split('.')[0].split(':')[0].strip()
                # Remove any formatting like **CNN** or 🔹CNN
                import re
                model_name = re.sub(r'[*🔹\-#]', '', model_name).strip()
                
                # Validate the model name is in our allowed list
                if model_name not in allowed_models:
                    # Try to find a match in the response
                    for allowed in allowed_models:
                        if allowed.lower() in auto_model_response.lower():
                            model_name = allowed
                            break
                    else:
                        model_name = "CNN"  # Default fallback
                
                st.subheader("🎯 Agent's Model Recommendation:")
                st.success(f"**Selected Model: {model_name}**")
                
                # Update training data to include just the model name
                training_data += f"Agent's recommended model: {model_name}\n"

            st.session_state.messages.append({"role": "user", "content": training_data})
            with st.chat_message("assistant"):
                try:
                    with st.spinner("⚙️ Training the model..."):
                        response = asyncio.run(chat_with_agent(st.session_state.messages))
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Execute the training code and display results
                    training_result = ToolWrapper.model_training(response)
                    
                    # Display the extracted Python code before training
                    if isinstance(training_result, dict):
                        st.subheader("Generated Python Training Code:")
                        st.code(training_result["extracted_code"], language="python")
                        
                        st.success("Model training has started on CPU due to GPU limitations!")
                        
                        # Now execute the training code
                        with st.spinner("⚙️ Executing training code..."):
                            execution_result = ToolWrapper.execute_training_code()
                        
                        # Display execution output
                        if execution_result["stdout"]:
                            st.subheader("📋 Training Output:")
                            st.text(execution_result["stdout"])
                    else:
                        # Fallback for old format
                        st.markdown(training_result)

                    model_files = glob.glob("*.keras") + glob.glob("*.h5") + glob.glob("*.pb") + \
                                glob.glob("*.pt") + glob.glob("*.pkl") + glob.glob("*.sav")
                    
                    # Store model files in session state to persist after download button clicks
                    st.session_state.trained_models = model_files
                    
                    if model_files:
                        # Sort by modification time and get the latest file
                        latest_model = max(model_files, key=lambda x: os.path.getmtime(x))
                        file_size = os.path.getsize(latest_model) / (1024*1024)  # Size in MB
                        
                        st.success(f"✅ Latest trained model:")
                        st.info(f"📁 {latest_model} ({file_size:.2f} MB)")
                        
                        try:
                            with open(latest_model, "rb") as f:
                                model_data = f.read()
                                st.download_button(
                                    label=f"⬇️ Download {os.path.basename(latest_model)}",
                                    data=model_data,
                                    file_name=os.path.basename(latest_model),
                                    mime="application/octet-stream",
                                    key=f"download_latest_{latest_model}"
                                )
                        except Exception as e:
                            st.error(f"Error reading {latest_model}: {str(e)}")
                    else:
                        st.warning("No model file found after training.")
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")

    # Show available models section (persists even after button clicks)
    st.subheader("📁 Latest Trained Model")
    available_models = glob.glob("*.keras") + glob.glob("*.h5") + glob.glob("*.pb") + \
                      glob.glob("*.pt") + glob.glob("*.pkl") + glob.glob("*.sav")
    
    if available_models:
        # Get the latest model based on modification time
        latest_model = max(available_models, key=lambda x: os.path.getmtime(x))
        
        col_info, col_download = st.columns([2, 1])
        
        with col_info:
            file_size = os.path.getsize(latest_model) / (1024*1024)  # Size in MB
            file_time = os.path.getmtime(latest_model)  # Use modification time
            st.write(f"**{latest_model}**")
            st.caption(f"Size: {file_size:.2f} MB | Modified: {time.ctime(file_time)}")
        
        with col_download:
            try:
                with open(latest_model, "rb") as f:
                    model_data = f.read()
                    st.download_button(
                        label="⬇️ Download",
                        data=model_data,
                        file_name=os.path.basename(latest_model),
                        mime="application/octet-stream",
                        key=f"persistent_download_latest_{latest_model.replace('.', '_')}"
                    )
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info("No trained models found. Train a model to see it here.")

    st.divider()

    st.header("🧪 Click to Test Your Model")

    with st.expander("Open to upload and test your image", expanded=False):

        # --- Upload an image section ---
        st.subheader("🖼️ Test Your Model Now")
        uploaded_image = st.file_uploader("Upload an image to predict", type=["png", "jpg", "jpeg"])

        if uploaded_image:
            st.session_state.uploaded_image = uploaded_image

        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, caption="Uploaded Image", width=200)

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

# --- Sidebar (Right side) ---
with st.sidebar:
    # --- Agent Chat Log ---
    st.subheader("💬 Agent Chat Log")
   
    if st.button("Clear Chat", type="primary"):
            st.session_state.messages = []
            st.session_state.uploaded_image = None
            st.session_state.last_prompt = None
            st.session_state.last_response = None
            st.session_state.last_powershell_command = None

    if st.button("🔄 Regenerate Response"):
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages.pop()

        # st.session_state.messages.append({"role": "user", "content": st.session_state.last_prompt})

        with st.chat_message("user"):
            st.markdown(st.session_state.last_prompt)

        with st.chat_message("assistant"):
            try:
                new_response = asyncio.run(chat_with_agent(st.session_state.messages))
                # st.session_state.messages.append({"role": "assistant", "content": new_response})
                st.session_state.last_response = new_response
                st.markdown(new_response)

            except Exception as e:
                new_response = f"⚠️ Error regenerating response: {str(e)}"
                # st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.session_state.last_response = None
                st.error(new_response)
        user = [msg for idx, msg in enumerate(reversed(st.session_state.messages)) if idx % 2 == 0]
        assistant = [msg for idx, msg in enumerate(reversed(st.session_state.messages)) if idx % 2 == 1]

        userAssistantPair = zip(user, assistant)

        if st.session_state.messages:
            for userMsg, assistantMsg in userAssistantPair:
                # print(msg)
                with st.chat_message(assistantMsg["role"]):
                    st.markdown(assistantMsg["content"])
                with st.chat_message(userMsg["role"]):
                    st.markdown(userMsg["content"])

        st.session_state.messages.append({"role": "user", "content": st.session_state.last_prompt})
        st.session_state.messages.append({"role": "assistant", "content": new_response})
    
    # --- Chat Input ---
    if prompt := st.chat_input("Your message"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.last_prompt = prompt

        messages = [msg for msg in st.session_state.messages]
        messages.append({"role": "user", "content": prompt})

        try:
            response = asyncio.run(chat_with_agent(messages))
            # st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.last_response = response
            with st.chat_message("assistant"):
                st.markdown(response)
        except Exception as e:
            response = f"⚠️ Error generating response: {str(e)}"
            # st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.last_response = None
            with st.chat_message("assistant"):
                st.error(response)
        
        user = [msg for idx, msg in enumerate(reversed(st.session_state.messages)) if idx % 2 == 0]
        assistant = [msg for idx, msg in enumerate(reversed(st.session_state.messages)) if idx % 2 == 1]

        userAssistantPair = zip(user, assistant)

        if st.session_state.messages:
            for userMsg, assistantMsg in userAssistantPair:
                # print(msg)
                with st.chat_message(assistantMsg["role"]):
                    st.markdown(assistantMsg["content"])
                with st.chat_message(userMsg["role"]):
                    st.markdown(userMsg["content"])

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})
