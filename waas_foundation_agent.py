from tool_wrapper import ToolWrapper
import os
import dotenv
import json
import re

# import user_proxy_agent

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core import RoutedAgent
from autogen_agentchat.messages import HandoffMessage

# Global variables
retriever = None
agent = None
system_message = None

from typing import AsyncGenerator, Sequence
import json
import re
from autogen_core import CancellationToken
from autogen_agentchat.messages import BaseMessage
from autogen_agentchat.messages import (
    AgentEvent,
    BaseChatMessage,
    ChatMessage,
    ModelClientStreamingChunkEvent,
    TextMessage,
)

from autogen_agentchat.base import Response


class WaaSFoundationAgent(AssistantAgent):
    # async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
    #     # Calls the on_messages_stream.
    #     response: Response | None = None
    #     async for message in self.on_messages_stream(messages, cancellation_token):
    #         if isinstance(message, Response):
    #             response = message
    #     assert response is not None
    #     return response

    # async def on_messages_stream(
    #     self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    # ):
    #     # print("Messages", messages)
    #     if messages:
    #        retrieved_doc = ToolWrapper.rag_retriever_tool(messages[-1].content)
    #        retrieved_doc_message = TextMessage(content=str(retrieved_doc), source="User")
    #        messages.append(retrieved_doc_message)
    #     #    print("updated messages", messages)
    #     #    print("Override function called\n")
    #     else:
    #        retrieved_doc = None  # Or handle it in another appropriate way

    #     async for response in super().on_messages_stream(messages, cancellation_token):
    #         yield response  # Forward each yielded response

    # await super().on_messages_stream(self, messages, cancellation_token )

    # You can add custom logic here if needed
    # yield Response(chat_message=ChatMessage(content="Custom Final Response!", source=self.name))

    def get_agent(model_client, tools=[]):
        system_prompt = """
        You are an intelligent AI assistant that generates Python code to train machine learning models using datasets hosted on GitHub.

        The user will provide:

        Task description: (e.g., train model with weather data, spam classification)

        GitHub Dataset URL: Can be a repo, raw file, or ZIP

        Selected data options: Choose among Training, Testing, Validation

        Training percentage: (e.g., 75%)

        Testing + Validation percentage: Remaining split (e.g., 25%)

        Model chosen: One or more of [CNN, RNN, SVM, Random Forest], or leave it blank to auto-infer

        Your Responsibilities
        
        
        Download the dataset:

        If repo: Clone it

        If ZIP: Download and extract

        If raw file: Download directly

        Support .csv, .xlsx, .txt, and image folders

        Inspect and load dataset:

        Auto-detect format (CSV, XLSX, image, text)

        Dynamically infer column names and number of features

        If loading CSV, print actual column names first to avoid mismatch errors

        Do not hardcode column names – infer label and feature columns based on content (e.g., label column with only two values)

        Preprocess the data:

        For structured/tabular data: handle nulls, encode categorical variables

        For text: clean, tokenize

        For images: resize, normalize

        Dynamically apply label encoding or one-hot encoding

        Normalize numeric features

        Split the dataset:

        Respect user-defined training percentage

        Split remaining into test and validation evenly, unless user gives explicit split

        Shuffle before splitting

        Model selection and training:

        Use user-selected model if specified

        If multiple model options (e.g., SVM or Random Forest), choose both or best suited based on data shape

        If unspecified, infer model type:

        CNN for images

        RNN/SVM for text

        Random Forest/SVM for structured/tabular

        Evaluation:

        For classification: Accuracy, F1 Score, Confusion Matrix

        For regression: MAE, RMSE, R²

        Print evaluation on both validation and test sets

        Error Handling:

        If dataset column count mismatches expectations, print column names and explain expected structure

        Catch and explain errors such as file not found, unzip failure, missing values, etc.

        Ensure all downloads work even if raw file URLs have ?raw=true issues

        Output:

        Only generate clean, runnable Python code

        Fully commented for readability

        Ready to run in Jupyter or Google Colab

        Avoid hardcoded paths or credentials

        Example Input (from user)
        
        Task description: train model with weather data  
        GitHub Dataset URL: https://github.com/Vaibhav-Mehta-19/linear-regression-weather-dataset/blob/master/weather.csv  
        Selected data options: Training, Testing, Validation  
        Training percentage: 75%  
        Testing + Validation percentage: 25%  
        Model chosen: Random Forest  
        
        Expected Output
        
        
        Python code that:

        Downloads the CSV

        Auto-loads and inspects the columns

        Splits into 75/12.5/12.5 (train/val/test)

        Preprocesses tabular data

        Trains a Random Forest model

        Evaluates with MAE, RMSE, R²

        Explains any assumptions with comments
        """

        # print("tools: ", tools)
        # Initialize Assistant Agent
        waas_foundation_agent = WaaSFoundationAgent(
            name="WorkloadsAssistant",
            description="An agent for answering the questions related to design of workload, workload definition and VI. Don't answer quality checks or health related queries.",
            tools=tools,
            model_client=model_client,
            system_message=system_prompt,
            handoffs=["user"],
            reflect_on_tool_use=True,
        )

        return waas_foundation_agent

    @staticmethod
    def initialize_waas_agent():
        global retriever, agent, system_message

        dotenv.load_dotenv()
        # Azure OpenAI Deployment Configurations
        deployment_name = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
        azure_openai_api_version = "2024-05-01-preview"
        azure_openai_api_key = os.getenv(
            "Observability_AZ_OPENAI_API_KEY", "a60ecf15e9484a70bae7fcaf1e6830aa"
        )
        azure_openai_endpoint = os.getenv(
            "ENDPOINT_URL", "https://yukta-workloaddefintion-openai.openai.azure.com/"
        )
        service_name = os.getenv(
            "AZURE_AI_SEARCH_SERVICE_NAME", "acsscognitivesearchhack"
        )

        # Initialize Assistant Agent
        waas_agent = AssistantAgent(
            name="WorkloadsAssistant",
            system_message="""
            You are an Azure solution architect focused on designing workloads for Azure and onboarding them to WaaS. 
            Ask users about their requirements before designing an architecture and always suggest a highly available, secure, 
            and reliable solution with at least 30 Azure resources. 
            If designing a workload, return a mermaid architecture diagram using tool "MermaidDiagramGenerator". Provide detailed reasoning for each resource selection. 
            Guide users step-by-step through WaaS onboarding and quality checks. 
            """,
            llm_config={
                "config_list": [
                    {
                        "model": deployment_name,
                        "api_key": azure_openai_api_key,
                        "api_type": "azure",
                        "api_version": azure_openai_api_version,
                    }
                ],
                "temperature": 0.02,
                "max_tokens": 2500,
            },
        )

        # Register tools
        # for tool_name, tool in ToolWrapper.get_tools().items():
        #     waas_agent.register_for_llm(
        #         name=tool["name"],
        #         description=tool["description"]
        #     )(tool["func"])
        #     waas_agent.register_for_execution(
        #         name=tool["name"],
        #         description=tool["description"]
        #     )(tool["func"])

        # print("Tools registered:", ToolWrapper.get_tools())
        waas_agent.register_for_llm(
            name="MermaidDiagramGenerator",
            description="Generates a diagram whenever the response is in Mermaid code.",
        )(ToolWrapper.generate_mermaid_html())
        return waas_agent

        waas_agent = initialize()
        messages = [
            f"System: {system_message.content}"
        ]  # Use list instead of string  # Add system message as the first entry

        # Convert conversation history into formatted messages
        for message in user_input:
            role = message.get(
                "role", ""
            ).capitalize()  # Ensure proper casing (User/Assistant)
            content = message.get("content", "")
            messages.append(f"{role}: {content}")

        # Extract latest user message for context retrieval
        latest_user_message = user_input[-1].get("content", "")

        # Retrieve relevant documents based on the latest user query
        docs = retrieve_from_azure_search(latest_user_message)

        # Append retrieved documents as context
        if docs:
            doc_content = "\n".join([doc.page_content for doc in docs])
            messages.append(f"Context:\n{doc_content}")

        # Convert messages list into a single formatted string
        messages_string = "\n".join(messages)

        # Now invoke the agent
        response = waas_agent.invoke(messages_string)
        print("💡 Response:", response)
        # Extract intermediate steps to check tool execution
        intermediate_steps = response.get("intermediate_steps", [])

        tool_responses = []
        for step in intermediate_steps:
            if (
                isinstance(step, tuple) and len(step) == 2
            ):  # Ensure it's a tuple of (AgentActionMessageLog, dict)
                _, response_dict = step  # Unpack tuple
                tool_name = response_dict.get("tool_name", "Unknown Tool")
                tool_message = response_dict.get("response", "No response received.")
                tool_responses.append(f"[{tool_name}]: {tool_message}")

        final_output = (
            "\n".join(tool_responses)
            if tool_responses
            else response.get("output", "No response generated.")
        )

        print("Final Response:", final_output)
        return final_output
