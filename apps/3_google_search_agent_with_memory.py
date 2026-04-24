# assign langgraph memory to the agent

from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

llm = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()
memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt = """You are an AI agent.
You MUST use Google Search tool for any question about current events, recent facts, or real-world information.
Do NOT answer from your own knowledge.
Always search first before answering.""",
checkpointer=memory
)

while True:
    query = input("User: ")
    if query.lower() == "quit":
        print("good bye")
    
    response = agent.invoke({"messages":[{"role":"user","content":query}]},{"configurable":{"thread_id":"1"}})
    print(response["messages"][-1].content)