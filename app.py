import streamlit as st
# import streamlit_mermaid as stmd
import os
import dotenv
import uuid
from pathlib import Path
import asyncio
import re
from swarm_team import chat_with_agent
import time
import random
from tool_wrapper import ToolWrapper
from PIL import Image
import glob

# Load environment variables
dotenv.load_dotenv()
# Set Streamlit page configuration
st.set_page_config(
    page_title="MLE Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.html("""<h2 style="text-align: center;"> <i> MLE Agent </i> </h2>""")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "last_powershell_command" not in st.session_state:
    st.session_state.last_powershell_command = ""
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "do_print" not in st.session_state:
    st.session_state.do_print = False

# Sidebar for API tokens and model selection
with st.sidebar:
    az_openai_api_key = os.getenv("AZ_OPENAI_API_KEY")
    st.session_state.az_openai_api_key = az_openai_api_key
    st.divider()
    models = ["azure-openai/gpt-4o"]
    st.selectbox("🤖 Select a Model", options=models, key="model")

    st.button(
        "Clear Chat",
        on_click=lambda: (
            st.session_state.messages.clear(),
            st.session_state.update({
                "last_powershell_command": "",
                "show_chat": False
            })
        ),
        type="primary"
    )

# Display chat history
chat_container = st.container()
if st.session_state.show_chat:
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# Function to extract PowerShell command
def extract_powershell_command(response):
    if "```powershell" in response:
        return response.split("```powershell")[1].split("```")[0].strip()
    return None

# Function to attempt Mermaid rendering with retry logic
def render_mermaid_with_retry(response, retries=3, delay=2):
    attempt = 0
    while attempt < retries:
        try:
            if "```mermaid" in response:
                mermaid_code = response.split("```mermaid")[1].split("```")[0].strip()
                if not is_valid_mermaid_code(mermaid_code):
                    raise ValueError("Mermaid syntax is invalid.")
                stmd.st_mermaid(mermaid_code)
            return True
        except Exception:
            attempt += 1
            if attempt >= retries:
                st.session_state.messages.append({"role": "user", "content": "⚠️ Mermaid syntax is incorrect. Please fix it."})
                response = asyncio.run(chat_with_agent(st.session_state.messages))
                st.session_state.last_response = response
                return False
            time.sleep(delay)
    return False

# Validate Mermaid syntax
def is_valid_mermaid_code(mermaid_code):
    return bool(re.match(r"graph\\s+[LRBT]", mermaid_code))

# Train model
training_data = ''

def train_model(task_desc, github_url, options, train_percent, test_val_percent, model_choice):
    global training_data
    training_data = (
        f"Task description: {task_desc}\n"
        f"GitHub Dataset URL: {github_url}\n"
        f"Selected data options: {', '.join(options)}\n"
        f"Training percentage: {train_percent}%\n"
        f"Testing + Validation percentage: {test_val_percent}%\n"
        f"Model chosen: {model_choice}\n"
    )

    st.write("Training data string:")
    st.text(training_data)

    # st.session_state.show_chat = True
    st.session_state.messages.append({"role": "user", "content": training_data})

    with st.chat_message("assistant"):
        try:
            response = asyncio.run(chat_with_agent(st.session_state.messages))
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.last_response = response
            st.session_state.do_print = True
            # st.success("Generating Code...!!!")

            if "```mermaid" in response:
                mermaid_code = response.split("```mermaid")[1].split("```")[0].strip()
                stmd.st_mermaid(mermaid_code)

            powershell_command = extract_powershell_command(response)
            st.session_state.last_powershell_command = powershell_command

            st.markdown(response)
            
            code_block_match = re.search(r"```python(.*?)```", response, re.DOTALL)
    
            # Check if a match is found
            if code_block_match:
                # Extract the code from the match and strip any leading/trailing whitespace
                training_data = code_block_match.group(1).strip()
                return training_data
            else:
                raise ValueError("No Python code block found in the response string")
            

        except Exception as e:
            error_message = f"⚠️ Error generating response: {str(e)}"
            # traceback.print_exc()
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.session_state.last_response = None
            st.error(error_message)

text_input = st.text_input("Enter the task description", "train model")
github_url = st.text_input("Enter GitHub dataset link", "")
options = st.multiselect("Select the data options (You can choose multiple)", ["Training", "Testing", "Validation"])
train_test_split = st.slider("Select the percentage split for training vs testing and validation", 10, 90, 70, 5)
test_val_percentage = 100 - train_test_split
st.write(f"Training data: {train_test_split}%")
st.write(f"Testing + Validation data: {test_val_percentage}%")
model = st.selectbox("Choose a model", options=["CNN", "RNN", "SVM", "Random Forest", "Auto Model"], index=4)

run_button = st.button("Run")
if run_button:
    train_model(text_input, github_url, options, train_test_split, test_val_percentage, model)
    print("do_print - 1: ", st.session_state.do_print)
    if st.session_state.do_print:
        st.success("Model training has started on CPU due to limitation of GPU server!")
    # st.write("There are 19988 images in the dataset.")
    # time.sleep(1)
    # st.write("The dataset is split into 3 folders: train, test, and validation.")
    # time.sleep(1)
    # st.write("The training data is 70% of the dataset, and the testing and validation data is 30%.")
    # time.sleep(1)
    # st.write("Training data is 13991 images, testing data is 2999 images, and validation data is 2998 images.")
    # time.sleep(20)
    
    st.markdown(ToolWrapper.model_training(training_data))

    # Look for any common model file extensions
    model_files = glob.glob("*.keras") + glob.glob("*.h5") + glob.glob("*.pb") + glob.glob("*.pt") + glob.glob("*.pkl") + glob.glob("*.sav")

    if model_files:
        model_path = model_files[0]  # Take the first found model
        st.success(f"Found trained model: `{model_path}`")

        with open(model_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Trained Model",
                data=f,
                file_name=os.path.basename(model_path),
                mime="application/octet-stream"
            )
    else:
        st.warning("No trained model file found (e.g., .keras, .h5, .pt, .pb, etc).")
    st.session_state.do_print = False
    # st.write("The model is trained using the CNN algorithm.")
    # st.write("Training Accuracy: 93%")
    # st.success("Model Trained.")
    # st.success("Model Testing Started on MAIA.")
    st.session_state.show_chat = True

    # Set folder path
    # cat_folder = Path(r"C:\Users\pchanchlani\Downloads\WorkloadsAgentAutogen\matched_images\cat")
    # dog_folder = Path(r"C:\Users\pchanchlani\Downloads\WorkloadsAgentAutogen\matched_images\dog")
    # incorrect_folder = Path(r"C:\Users\pchanchlani\Downloads\WorkloadsAgentAutogen\matched_images\incorrect")
    
    # # Step 1: Initialize session state only once
    # if "tagged_images" not in st.session_state:
        
    #     tagged_images = []
    #     # Load images and label them
    #     for img in cat_folder.glob("*.jpg"):
    #         tagged_images.append(("Cat", img))
    #     for img in dog_folder.glob("*.jpg"):
    #         tagged_images.append(("Dog", img))
    #     for img in incorrect_folder.glob("*.jpg"):
    #         tagged_images.append(("Incorrect", img))

    #     # Shuffle and store in session
    #     random.shuffle(tagged_images)
    #     st.session_state.tagged_images = tagged_images[:32]  # Limit to first 32 if more
    #     st.session_state.current_index = 0
    #     st.session_state.total_images = len(st.session_state.tagged_images)

    # # Placeholder for image display
    # image_placeholder = st.empty()
    # text_placeholder = st.empty()


    # num_images = len(st.session_state.tagged_images)

    # # Loop through the images and display them in batches of 4 per row
    # for i in range(0, num_images, 11):
    #     # Create 4 columns for the current row (or fewer if the remaining images are less than 4)
    #     cols = st.columns(11)

    #     # Display up to 4 images in the current row
    #     for j in range(11):
    #         index = i + j
    #         if index < num_images:
    #             tag, img_path = st.session_state.tagged_images[index]
    #             if tag == 'Cat' or tag == 'Dog':
    #                 caption = (
    #                     f'<p style="color: green; font-size: 18px; '
    #                     f'padding-left: 45px; "><strong>Class: {tag}</strong></p>'
    #                 )
    #             else:
    #                 caption = (
    #                     f'<p style="color: red; font-size: 18px; '
    #                     f'padding-left: 35px; "><strong>Class: {tag}</strong></p>'
    #                 )

                
    #             # Open and display the image using its path
    #             cols[j].image(Image.open(img_path), caption="", width=180)
    #             cols[j].markdown(caption, unsafe_allow_html=True)  # Allow HTML in caption
    #             time.sleep(2)
    
    # st.success("Model Tested on MAIA.")

    

# # Chat input only after training has started
# if st.session_state.show_chat and (prompt := st.chat_input("Your message")):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     st.session_state.last_prompt = prompt

#     with st.chat_message("assistant"):
#         try:
#             response = asyncio.run(chat_with_agent(st.session_state.messages))
#             st.session_state.messages.append({"role": "assistant", "content": response})
#             st.session_state.last_response = response

#             if "```mermaid" in response:
#                 mermaid_code = response.split("```mermaid")[1].split("```")[0].strip()
#                 stmd.st_mermaid(mermaid_code)

#             powershell_command = extract_powershell_command(response)
#             st.session_state.last_powershell_command = powershell_command

#             st.markdown(response)
#         except Exception as e:
#             error_message = f"⚠️ Error generating response: {str(e)}"
#             traceback.print_exc()
#             st.session_state.messages.append({"role": "assistant", "content": error_message})
#             st.session_state.last_response = None
#             st.error(error_message)

# # Execute PowerShell command
# if st.session_state.last_powershell_command:
#     st.markdown("### Detected PowerShell Command:")
#     st.code(st.session_state.last_powershell_command, language="powershell")

#     if st.button("⚡ Create Virtual Instance on portal"):
#         st.session_state.messages.append({"role": "user", "content": "Create virtual instance"})
#         with st.chat_message("assistant"):
#             try:
#                 response = asyncio.run(chat_with_agent(st.session_state.messages))
#                 st.session_state.messages.append({"role": "assistant", "content": response})
#                 st.session_state.last_response = response

#                 if "```mermaid" in response:
#                     if not render_mermaid_with_retry(response):
#                         st.session_state.messages.append({"role": "assistant", "content": "⚠️ Mermaid syntax was incorrect, reprocessing the request."})

#                 st.markdown(response)
#             except Exception as e:
#                 error_message = f"⚠️ Error generating response: {str(e)}"
#                 traceback.print_exc()
#                 st.session_state.messages.append({"role": "assistant", "content": error_message})
#                 st.session_state.last_response = None
#                 st.error(error_message)

# # Regenerate response
# if st.session_state.get("last_prompt"):
#     if st.button("🔄 Regenerate Response"):
#         if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
#             st.session_state.messages.pop()

#         st.session_state.messages.append({"role": "user", "content": st.session_state.last_prompt})

#         with st.chat_message("user"):
#             st.markdown(st.session_state.last_prompt)

#         with st.chat_message("assistant"):
#             try:
#                 new_response = asyncio.run(chat_with_agent(st.session_state.messages))
#                 st.session_state.messages.append({"role": "assistant", "content": new_response})
#                 st.session_state.last_response = new_response

                
#                 st.markdown(new_response)
#             except Exception as e:
#                 error_message = f"⚠️ Error generating response: {str(e)}"
#                 traceback.print_exc()
#                 st.session_state.messages.append({"role": "assistant", "content": error_message})
#                 st.session_state.last_response = None
#                 st.error(error_message)
