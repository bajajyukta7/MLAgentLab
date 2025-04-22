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
        result = subprocess.run([sys.executable, "generated_model.py"], capture_output=True, text=True)
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        # accuracy_matches = re.findall(r'accuracy:\s+([0-9.]+)', result.stdout)
        # training_accuracies = [float(acc) for acc in accuracy_matches]
        return f"```bash\n{result.stdout}\n```"
       
    @staticmethod
    def get_model_training_tool():
        model_training_tool = FunctionTool(
            ToolWrapper.model_training,
            description="Fetch the relevant documents for a user query from RAG database.",
        )
        return model_training_tool
    
    @staticmethod
    def model_test(model_training_code:str) -> str :
        # print("Training model with code:\n", model_training_code)
        with open("test_model_code.py", "w") as f:
            f.write(model_training_code)
        result = subprocess.run([sys.executable, "test_model_code.py"], capture_output=True, text=True)
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        accuracy_matches = re.findall(r'accuracy:\s+([0-9.]+)', result.stdout)
        training_accuracies = [float(acc) for acc in accuracy_matches]
        return training_accuracies
    
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