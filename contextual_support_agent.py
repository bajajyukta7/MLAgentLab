# import user_proxy_agent
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent


    
@staticmethod
def get_agent(model_client,tools=[]):
    # Initialize Assistant Agent
    observability_agent = AssistantAgent(
        name="ContextualSupportAssistant",
        # description="An agent for answering health and observability related queries.",
        description="for health",
        tools=tools,
        model_client=model_client,
        system_message="""
        """,
        handoffs=["user"],
        reflect_on_tool_use = True
    )
    
    return observability_agent