import streamlit as st
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper, SearxSearchWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun, SearxSearchRun
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
import os


load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


st.set_page_config(page_title="🔍 AI Web Search Assistant", layout="wide")
st.title("🌐 Web Search Assistant with AI + LangChain")

temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.7, 0.1)
st.sidebar.markdown("Built with 💬 LangChain + Streamlit")

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
duck_wrapper = DuckDuckGoSearchAPIWrapper(max_results=1, source='news')
search_wrap = SearxSearchWrapper(searx_host="http://127.0.0.1:8888")

wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
duck = DuckDuckGoSearchRun(api_wrapper=duck_wrapper)
search = SearxSearchRun(wrapper=search_wrap, response_format='content_and_artifact')

tools = [wiki, duck, search]


model = ChatGroq(model="llama3-70b-8192", temperature=temperature)


prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with memory capabilities. You can answer questions using web tools like Wikipedia, DuckDuckGo, or Searx when necessary. Always retain and recall user-provided information, such as names or preferences, when asked. For example, if a user tells you their name, store it and provide it when they ask, 'What is my name?'"""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}
session_id = "user-session"
memory = st.session_state["chat_history"].setdefault(session_id, InMemoryChatMessageHistory())

agent_with_chat_history = RunnableWithMessageHistory(
    AgentExecutor(agent=create_tool_calling_agent(model, tools, prompt), tools=tools, verbose=True),
    lambda _: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm your AI assistant. Ask me anything, or tell me something like your name, and I'll remember it for you."}
    ]


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if user_prompt := st.chat_input(placeholder="Type your question here..."):
    st.chat_message("user").markdown(user_prompt)
    st.session_state["messages"].append({"role": "user", "content": user_prompt})

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        try:
            response = agent_with_chat_history.invoke(
                {"input": user_prompt},
                config={
                    "configurable": {"session_id": session_id},
                    "callbacks": [st_cb]
                }
            )
            output = response["output"]
        except Exception as e:
            output = f"❌ Error: {e}"

        st.markdown(output)
        st.session_state["messages"].append({"role": "assistant", "content": output})