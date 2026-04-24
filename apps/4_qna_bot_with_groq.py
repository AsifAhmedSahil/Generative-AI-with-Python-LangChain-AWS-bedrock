# requirement - env,llm,google search- serper, agent,tools,streaming,streamlit,memory

from dotenv import load_dotenv
load_dotenv()


from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
 
import streamlit as st

llm = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()
tools=[search.run]

memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt = """You are an AI agent.
You MUST use Google Search tool for any question about current events, recent facts, or real-world information.
Do NOT answer from your own knowledge.
Always search first before answering.""",
checkpointer=memory
)

query = "Who is the pm of bd?"

response = agent.invoke({"messages":[{"role":"user","content":query}]},{"configurable":{"thread_id":"1"}})
print(response["messages"][-1].content)

