from training_agent import MLEAgent
from inference_agent import InferenceAgent
import selector_agent
import os
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from tool_wrapper import ToolWrapper

# Utility class for agents
class AgentWrapper:
    
    # This function can be moved to a base class from where agents will inherit.
    @staticmethod
    def get_model_client():
        
        
        # Azure OpenAI Deployment Configurations
        deployment_name = os.getenv("DEPLOYMENT_NAME", "gpt-4o")  
        azure_openai_api_version = "2024-05-01-preview"
        azure_openai_api_key = os.getenv("Observability_AZ_OPENAI_API_KEY")  
        azure_openai_endpoint = os.getenv("ENDPOINT_URL", "https://yukta-workloaddefintion-openai.openai.azure.com/")  
        service_name = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME", "acsscognitivesearchhack") 
        
        az_model_client = AzureOpenAIChatCompletionClient(
            azure_deployment=deployment_name,
            model=deployment_name,
            api_version=azure_openai_api_version,
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_api_key # For key-based authentication.
        )
        return az_model_client
        
    @staticmethod
    def get_agents():
        """Ensures tools are registered and returns them."""
        
        return [selector_agent.get_agent(model_client=AgentWrapper.get_model_client(),
                                              tools=[]),
                MLEAgent.get_agent(model_client=AgentWrapper.get_model_client(),
                                                tools=[ToolWrapper.get_model_training_tool]),
                InferenceAgent.get_agent(model_client=AgentWrapper.get_model_client(),
                                                tools=[ToolWrapper.get_model_test_tool])]