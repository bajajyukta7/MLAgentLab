# import streamlit_mermaid as stmd
import streamlit as st
import re
import sys
import subprocess
from autogen_core.tools import FunctionTool
import os
import glob
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
# import tensorflow as tf  # Remove this import from top level
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
        # Extract Python code from the response (remove formatting and get code blocks)
        print("Raw response received:\n", model_training_code)
        
        # Try to extract Python code from markdown code blocks
        python_code_pattern = r'```(?:python|py)?\s*(.*?)```'
        code_matches = re.findall(python_code_pattern, model_training_code, re.DOTALL)
        
        if code_matches:
            # Use the first (or largest) code block found
            extracted_code = max(code_matches, key=len).strip()
            print("Extracted Python code:\n", extracted_code)
        else:
            # If no code blocks found, try to extract lines that look like Python code
            lines = model_training_code.split('\n')
            python_lines = []
            for line in lines:
                # Skip lines that are obviously not Python code
                stripped_line = line.strip()
                if (stripped_line and 
                    not stripped_line.startswith('**🔹') and 
                    not stripped_line.startswith('🛠') and
                    not stripped_line.startswith('##') and
                    not stripped_line.startswith('###') and
                    not stripped_line.startswith('MLEAgent:') and
                    not stripped_line.startswith('InferenceAgent:') and
                    not stripped_line.startswith('SelectorAgent:') and
                    (stripped_line.startswith('import ') or 
                     stripped_line.startswith('from ') or
                     'import' in stripped_line or
                     '=' in stripped_line or
                     'def ' in stripped_line or
                     'class ' in stripped_line or
                     'if ' in stripped_line or
                     'for ' in stripped_line or
                     'while ' in stripped_line or
                     'with ' in stripped_line or
                     'try:' in stripped_line or
                     'except' in stripped_line or
                     stripped_line.endswith(':') or
                     'print(' in stripped_line or
                     'model.' in stripped_line or
                     '.fit(' in stripped_line or
                     '.compile(' in stripped_line or
                     '.evaluate(' in stripped_line or
                     '.predict(' in stripped_line or
                     'tf.' in stripped_line or
                     'keras.' in stripped_line or
                     'np.' in stripped_line)):
                    python_lines.append(line)
            
            if python_lines:
                extracted_code = '\n'.join(python_lines)
                print("Extracted Python code from lines:\n", extracted_code)
            else:
                # Fallback: use the original code as-is
                extracted_code = model_training_code
                print("No Python code extraction possible, using original:\n", extracted_code)
        
        # Write the extracted code to file
        with open("train_model_code.py", "w") as f:
            f.write(extracted_code)
        
        # Return the extracted code first (execution will be done separately)
        return {
            "extracted_code": extracted_code,
            "stdout": "",
            "stderr": "",
            "returncode": 0
        }
    
    @staticmethod
    def execute_training_code() -> dict:
        """Execute the previously extracted training code"""
        try:
            # Execute the Python code with better error handling
            result = subprocess.run([sys.executable, "train_model_code.py"], 
                                  capture_output=True, text=True, timeout=600)  # 10 minute timeout
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
            
            # Filter out common warnings that aren't critical
            filtered_stderr = []
            if result.stderr:
                for line in result.stderr.split('\n'):
                    if not any(warning in line.lower() for warning in [
                        'warning: failed to remove contents',
                        'notice] a new release of pip',
                        'warning:',
                        'you can safely remove it manually'
                    ]):
                        filtered_stderr.append(line)
            
            return {
                "stdout": result.stdout,
                "stderr": '\n'.join(filtered_stderr),
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Training timed out after 10 minutes. Please try with a smaller dataset or simpler model.",
                "returncode": 1
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Error executing training code: {str(e)}",
                "returncode": 1
            }
       
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
            # Lazy import TensorFlow to avoid startup issues
            try:
                import tensorflow as tf
            except ImportError as e:
                return f"Error: TensorFlow not properly installed. {str(e)}"
            except Exception as e:
                return f"Error: TensorFlow import failed. {str(e)}"
            
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

            # Look for model files in the current directory
            model_files = glob.glob("*.h5") + glob.glob("*.keras") + glob.glob("*cat*dog*.h5")
            if model_files:
                model_path = model_files[0]  # Use the first model file found
            else:
                # Fallback to common model names in current directory
                possible_paths = [
                    "cat_dog_classifier.h5",
                    "cat_dog_model.h5", 
                    "model.h5",
                    "best_model.h5"
                ]
                model_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                
                if model_path is None:
                    return "Error: No trained model found. Please train a model first."
            
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
    def resize_images(image_input) -> str:
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

        except Exception as e:
            print("Exception occurred!")
            traceback.print_exc()
            return f"Error during classification: {str(e)}"

    @staticmethod
    def get_resize_image_tool():
        retrieval_tool = FunctionTool(
            ToolWrapper.resize_images,
            description="Process the data for training and resize if available",
        )
        return retrieval_tool