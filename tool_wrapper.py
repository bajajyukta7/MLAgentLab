# import streamlit_mermaid as stmd
import streamlit as st
import re
import sys
import subprocess
from autogen_core.tools import FunctionTool
import os
from azure.search.documents import SearchClient
# import user_proxy_agent
from azure.core.credentials import AzureKeyCredential
import webbrowser
from pathlib import Path

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from typing import Union
from PIL import Image
import numpy as np
import tensorflow as tf
import traceback
import base64
import io



class ToolWrapper:
    """Manages multiple tools and provides dynamic tool selection."""
    
    @staticmethod
    def rag_retriever_tool(user_input:str) -> str :
        
        # search_endpoint = os.getenv("SEARCH_ENDPOINT", "https://acsscognitivesearchhack.search.windows.net")  
        search_key = search_key = os.getenv("AZ_OPENAI_API_SEARCH_KEY")   
        index_name = "mcwh28mar"
        service_name = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME", "acsscognitivesearchhack") 
        
        print("Retrieving from azure search for user input:\n", user_input)
        
        # messages = []
        # # Convert conversation history into formatted messages
        # for message in user_input:
        #     role = message.get("role", "").capitalize()  # Ensure proper casing (User/Assistant)
        #     content = message.get("content", "")
        #     messages.append(f"{role}: {content}")

        # # Extract latest user message for context retrieval
        # latest_user_message = user_input[-1].get("content", "")
        search_endpoint = os.getenv("SEARCH_ENDPOINT", "https://acsscognitivesearchhack.search.windows.net")  
        search_key = search_key = os.getenv("AZ_OPENAI_API_SEARCH_KEY")   
        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key)
        )

        docs = ""
        # docs = list(search_client.search(user_input))
        # print("docs:", docs)
        doc_content = "retriever data:"
        # Append retrieved documents as context
        if docs:
            doc_content = "\n".join([doc["content"] for doc in docs])
            # messages.append(f"Context:\n{doc_content}")

        # Convert messages list into a single formatted string
        # messages_string = "\n".join(messages)
        # return messages_string
        return doc_content
    
    @staticmethod
    def model_training(model_training_code:str) -> str :
        # print("Training model with code:\n", model_training_code)
        with open("train_model_code.py", "w") as f:
            f.write(model_training_code)
        result = subprocess.run([sys.executable, "train_model_code.py"], capture_output=True, text=True)
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        # accuracy_matches = re.findall(r'accuracy:\s+([0-9.]+)', result.stdout)
        # training_accuracies = [float(acc) for acc in accuracy_matches]
        return f"```bash\n{result.stdout}\n```"
       
    @staticmethod
    def get_model_training_tool():
        model_training_tool = FunctionTool(
            ToolWrapper.model_training,
            description="Train the model.",
        )
        return model_training_tool
    
    @staticmethod
    def model_test(image_input) -> str:
        try:
            print("Step 0: Entered model_test")

            img_height, img_width = 150, 150

            # 🔄 Convert base64 → PIL.Image if needed
            if isinstance(image_input, str):
                if image_input.strip().startswith("/9j"):  # likely base64 JPEG
                    print("Step 0.5: Decoding base64 string to image")
                    image_bytes = base64.b64decode(image_input)
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    print(f"Step 0.5: Opening image from file path: {image_input}")
                    image = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                raise ValueError("Unsupported input type for image")

            print("Step 1: Resizing image")
            image = image.resize((img_width, img_height))
            image = np.array(image).astype("float32")

            print(f"Step 2: Image shape after resize: {image.shape}")
            image = image / 255.0
            image = np.expand_dims(image, axis=0)
            print(f"Step 3: Image shape after expand_dims: {image.shape}")

            model_path = r"Q:\MLAgentLab\cat_dog_classifier.h5"
            print(f"Step 4: Loading model from {model_path}")
            model = tf.keras.models.load_model(model_path)

            print("Step 5: Model loaded, predicting...")
            prediction = model.predict(image)
            print(f"Step 6: Prediction result: {prediction}")

            label = "cat" if prediction[0][0] < 0.5 else "dog"
            print(f"Step 7: Classified as: {label}")
            return f"Classified as: {label}"

        except Exception as e:
            print("Exception occurred!")
            traceback.print_exc()
            return f"Error during classification: {str(e)}"
        
    @staticmethod
    def get_model_test_tool():
        model_test_tool = FunctionTool(
            ToolWrapper.model_test,
            description="Test the model.",
        )
        return model_test_tool
        
    # @staticmethod
    # def model_test(model_training_code:str) -> str :
    #     # Load model once (to avoid reloading every time the agent calls this)
    #     model = tf.keras.models.load_model('best_model.h5')
    #     img_height, img_width = 150, 150

    #     def classify_image_agent(image: Union[Image.Image, np.ndarray]) -> str:
    #         try:
    #             if isinstance(image, Image.Image):
    #                 image = image.resize((img_width, img_height))
    #                 image = np.array(image)

    #             image = image / 255.0
    #             image = np.expand_dims(image, axis=0)

    #             prediction = model.predict(image)
    #             label = "cat" if prediction[0][0] < 0.5 else "dog"
    #             return f"Classified as: {label}"
    #         except Exception as e:
    #             return f"Error during classification: {str(e)}"

    
    @staticmethod
    def get_retrieval_tool():
        retrieval_tool = FunctionTool(
            ToolWrapper.rag_retriever_tool,
            description="Fetch the relevant documents for a user query from RAG database.",
        )
        return retrieval_tool
    
    @staticmethod
    async def execute_arm_actions(script_content: str):
        """
        Executes a PowerShell script to perform Azure Resource Manager (ARM) actions.

        Parameters:
        - script_content (str): The PowerShell script content to execute.

        Returns:
        - str: The output from the PowerShell script execution.
        """
        
        # print("Execute powershell command")
        # # Open a web link
        # os.system("start https://ms.portal.azure.com/#browse/Microsoft.Workloads%2FsapVirtualInstances") 

        # def open_azure(request):
        url = "https://ms.portal.azure.com/#browse/Microsoft.Workloads%2FsapVirtualInstances"
        webbrowser.open(url)  # This opens the URL in the default browser.

        # work_dir = Path("coding")
        # work_dir.mkdir(exist_ok=True)

        # venv_dir = work_dir / ".ven2"
        # venv_builder = venv.EnvBuilder(with_pip=True)
        # venv_builder.create(venv_dir)
        # venv_context = venv_builder.ensure_directories(venv_dir)

        # local_executor = LocalCommandLineCodeExecutor(work_dir=work_dir, virtual_env_context=venv_context)
        # await local_executor.execute_code_blocks(
        #   code_blocks=[
        #      CodeBlock(language="python", code="pip install xyz"),
        # ])
        # cancellation_token=CancellationToken(),
        # try:
        #     # Execute the PowerShell script
        #     result = subprocess.run(
        #         ["powershell", "-Command", script_content],
        #         capture_output=True,
        #         text=True,
        #         check=True
        #     )
        #     return result.stdout
        # except subprocess.CalledProcessError as e:
        #     return f"An error occurred while executing the PowerShell script: {e.stderr}"

    @staticmethod
    def get_execution_tool():
        execution_tool = FunctionTool(
            ToolWrapper.execute_arm_actions,
            description="Execute powershell commands or register VI",
        )
        return execution_tool