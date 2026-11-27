import hmac
import re
from uuid import uuid4

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


@st.cache_resource
def get_agent():
    from agent.agent import Context, agent

    return agent, Context


def start_chat():
    st.session_state.thread_id = uuid4().hex
    st.session_state.messages = []
    st.session_state.turn_failed = False


def sign_out():
    del st.session_state.user_id
    start_chat()


def answer_prompt(prompt):
    agent, context_type = get_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": st.session_state.thread_id}},
        context=context_type(user_id=st.session_state.user_id),
    )
    message = result["messages"][-1]
    if message.type != "ai" or message.tool_calls:
        raise ValueError("The agent returned without a final answer")
    if isinstance(message.content, str):
        answer = message.content.strip()
    else:
        answer = "\n\n".join(
            block["text"] for block in message.content_blocks if block["type"] == "text"
        ).strip()
    if not answer:
        raise ValueError("The agent returned an empty answer")
    return answer


st.set_page_config(page_title="Grocery & Recipe Assistant", page_icon="🛒")
st.title("Grocery & Recipe Assistant")
st.caption("Plan meals, use what you have, and estimate your grocery bill.")

try:
    missing_sections = [
        section for section in ("kroger", "openai", "neon", "demo_users")
        if not st.secrets.get(section)
    ]
except StreamlitSecretNotFoundError:
    missing_sections = ["kroger", "openai", "neon", "demo_users"]

if missing_sections:
    st.info("App setup is required. Add the secrets shown in `.streamlit/secrets.toml.example`.")
    st.stop()

if "user_id" not in st.session_state:
    with st.form("sign_in", clear_on_submit=True):
        username = st.text_input("Username", help="Use letters, numbers, underscores, or hyphens.")
        access_code = st.text_input("Access code", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    st.caption("The assistant may save allergies, diet preferences, pantry items, and your usual budget to your profile.")
    if submitted:
        username = username.strip().lower()
        if not re.fullmatch(r"[a-z0-9_-]{1,32}", username):
            st.error("Enter a username with 1–32 letters, numbers, underscores, or hyphens.")
            st.stop()
        expected_code = st.secrets["demo_users"].get(username, "")
        if not expected_code or not hmac.compare_digest(access_code.encode(), expected_code.encode()):
            st.error("The username or access code is incorrect.")
            st.stop()
        st.session_state.user_id = username
        start_chat()
        st.rerun()
    st.stop()

with st.sidebar:
    st.markdown(f"Signed in as `{st.session_state.user_id}`")
    st.button("New conversation", on_click=start_chat)
    st.button("Sign out", on_click=sign_out)
    st.caption("A new conversation clears the chat and keeps your saved profile.")
    st.caption("Price estimates use a Kroger near Cincinnati (ZIP 45202) and exclude tax.")

if not st.session_state.messages:
    st.markdown("**Try asking**")
    st.markdown(
        '- “I have chicken, rice, and garlic. What can I make?”\n'
        '- “Plan two dinners for under $30 and give me a shopping list.”'
    )

for entry in st.session_state.messages:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

if st.session_state.turn_failed:
    st.error("The assistant couldn’t finish that request. Start a new conversation to try again.")

prompt = st.chat_input("Ask about recipes, ingredients, or grocery prices", disabled=st.session_state.turn_failed)
if prompt and prompt.strip():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Working on your request…"):
                answer = answer_prompt(prompt)
        except Exception as error:
            # The exception text may contain credentials or user data.
            print(f"Grocery assistant request failed ({type(error).__name__}).")
            st.session_state.turn_failed = True
            st.rerun()
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
