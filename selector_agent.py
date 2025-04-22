# import user_proxy_agent
from autogen_agentchat.agents import AssistantAgent
import streamlit as st

@staticmethod
def getInput():
    print("Getting input from user")
    st.rerun()

@staticmethod
def get_agent(model_client, tools=[]):
    
    selector_agent = AssistantAgent(
        name="SelectorAgent",
        description="An agent for routing the call to correct agent",
        tools=tools,
        model_client=model_client,
        system_message="""
        - You are a routing agent which sends the user input to agents and receives the response from agents.
        - After each step handoff to user with proper content. context woulb be all the last messages
        - Always consider the last input from user for routing the call. If last input is very generic then refer last 2-3.
        - Do not generate response on user's behalf
        - Coordinate by delegating to specialized agents:
        - MLEAgent: For designing workloads, creating Virtual instance or VI, or workload defintions
        - Always handoff to a single agent at a time.
        - Consolidate all the responses from the agents for immediate conversation and send a combined response to the user.
        - Strictly don't add any new context to message, just keep forwarding the messages.
        - Never return empty response to any agent or customer.
        """,
        handoffs=["MLEAgent", "InferenceAgent"],
        reflect_on_tool_use = True
    )
    
    return selector_agent
    
    return assisnt_routing_agent