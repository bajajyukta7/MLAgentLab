# import user_proxy_agent
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.messages import HandoffMessage
@staticmethod
def get_agent(model_client, tools=[]):
    
    # Initialize Assistant Agent
    config_validation_agent = AssistantAgent(
        name="ConfigValidationAssistant",
        description="An agent for answering config checks or quality checks related queries. Don't answer health related queries",
        tools=tools,
        model_client=model_client,
        system_message="""
        Always call the retrival tool to get data relevant to user query and use it as context.
        You are an assistant for workload quality check/config checks generation and executuion.
        Recommend quality checks based on the retrieved data.
        Return the quality checks template as response.
        Always handoff back to user when analysis is complete.
        """,
        handoffs=["user"],
        reflect_on_tool_use = True
    )
    
    return config_validation_agent