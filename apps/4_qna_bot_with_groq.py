# requirement - env,llm,google search- serper, agent,tools,streaming,streamlit,memory

from dotenv import load_dotenv
load_dotenv()


from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
 
import streamlit as st

llm = ChatGroq(model="openai/gpt-oss-20b",streaming=True)
search = GoogleSerperAPIWrapper()
tools=[search.run]

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []



agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt = """You are an AI agent.
You MUST use Google Search tool for any question about current events, recent facts, or real-world information.
Do NOT answer from your own knowledge.
Always search first before answering. Use tools carefully.
Only pass required arguments.
Do not pass null or extra fields.""",
checkpointer=st.session_state.memory
)
st.subheader("QuickChat - Answer your thoughts is seconds...")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything")


if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":query})
    response = agent.stream({"messages":[{"role":"user","content":query}]},{"configurable":{"thread_id":"1"}},stream_mode="messages")
    
    ai_container = st.chat_message("ai")

    with ai_container:
        space = st.empty()

        message = ""

        for chunk in response:
            message = message + chunk[0].content
            space.write(message)

        st.session_state.history.append({"role":"ai","content":message})
   
   


