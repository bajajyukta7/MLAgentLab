# import user_proxy_agent
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent


    
@staticmethod
def get_agent(model_client,tools=[]):
    # Initialize Assistant Agent
    observability_agent = AssistantAgent(
        name="ObservabilityHealthAssistant",
        description="An agent for answering workload health checks related queries.",
        tools=tools,
        model_client=model_client,
        system_message="""
        Use retrieved context for azure health and monitoring of workloads. 
        If no context is found, base your answer on prior knowledge."
        Answer the questions based on azure guidelines.
        Always handoff back to orchestator when analysis is complete.
        """,
        handoffs=["user"],
        reflect_on_tool_use = True
    )
    
    return observability_agent