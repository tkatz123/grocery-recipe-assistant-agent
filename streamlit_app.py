from hashlib import pbkdf2_hmac
import hmac
import re
from secrets import token_bytes
from uuid import uuid4

import psycopg
from psycopg_pool import ConnectionPool
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


USERNAME_PATTERN = r"[a-z0-9_-]{1,32}"
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 128
PASSWORD_ITERATIONS = 600_000
REQUIRED_SETTINGS = (
    ("kroger", "client_id"),
    ("kroger", "client_secret"),
    ("openai", "api_key"),
    ("neon", "database_url"),
)


@st.cache_resource
def get_agent():
    from agent.agent import Context, agent
    from tools.krogertool import get_client

    # Initialize the cached client in Streamlit before agent tools use it on worker threads.
    get_client()
    return agent, Context


@st.cache_resource
def get_account_pool():
    pool = ConnectionPool(
        st.secrets["neon"]["database_url"],
        min_size=0,
        max_size=4,
        timeout=8,
        kwargs={"connect_timeout": 5},
        open=True,
    )
    try:
        with pool.connection() as conn:
            # Serialize table creation if two app instances start together.
            conn.execute("SELECT pg_advisory_xact_lock(688, 9)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS grocery_assistant_accounts (
                    username TEXT PRIMARY KEY CHECK (username ~ '^[a-z0-9_-]{1,32}$'),
                    password_salt BYTEA NOT NULL,
                    password_hash BYTEA NOT NULL
                )
            """)
    except psycopg.Error:
        pool.close()
        raise
    return pool


def hash_password(password, salt):
    return pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)


def create_account(username, password, demo_users):
    if username in demo_users:
        return False
    salt = token_bytes(16)
    password_hash = hash_password(password, salt)
    with get_account_pool().connection() as conn:
        # Existing Neon profiles are tied to their old username, even if the
        # demo entry has since been removed. LangGraph joins namespace labels with dots.
        if conn.execute("SELECT to_regclass('store') IS NOT NULL").fetchone()[0]:
            if conn.execute(
                "SELECT 1 FROM store WHERE prefix = %s AND key = 'profile'",
                (f"users.{username}",),
            ).fetchone():
                return False
        return conn.execute(
            """
            INSERT INTO grocery_assistant_accounts (username, password_salt, password_hash)
            VALUES (%s, %s, %s)
            ON CONFLICT (username) DO NOTHING
            RETURNING username
            """,
            (username, salt, password_hash),
        ).fetchone() is not None


def valid_sign_in(username, password, demo_users):
    with get_account_pool().connection() as conn:
        account = conn.execute(
            "SELECT password_salt, password_hash FROM grocery_assistant_accounts WHERE username = %s",
            (username,),
        ).fetchone()
    if account is None and username in demo_users:
        code = demo_users[username]
        return isinstance(code, str) and bool(code) and hmac.compare_digest(
            password.encode("utf-8"), code.encode("utf-8")
        )
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        return False
    salt, expected_hash = account if account else (bytes(16), bytes(32))
    matches = hmac.compare_digest(hash_password(password, salt), expected_hash)
    return account is not None and matches


def show_account_form():
    demo_users = st.secrets.get("demo_users", {})
    action = st.radio("Account", ("Sign in", "Create account"), horizontal=True)
    creating = action == "Create account"
    if not creating and demo_users:
        st.caption("For a demo account, enter your access code in the password field.")
    with st.form("create_account" if creating else "sign_in", clear_on_submit=True):
        username = st.text_input("Username", help="Use 1–32 letters, numbers, underscores, or hyphens.")
        password = st.text_input(
            "Password",
            type="password",
            help=f"Use {MIN_PASSWORD_LENGTH}–{MAX_PASSWORD_LENGTH} characters." if creating else None,
        )
        if creating:
            confirmation = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button(action, type="primary")
    st.caption("The assistant may save allergies, diet preferences, pantry items, and your usual budget to your profile.")
    if not submitted:
        return

    username = username.strip().lower()
    if not re.fullmatch(USERNAME_PATTERN, username):
        st.error("Enter a username with 1–32 letters, numbers, underscores, or hyphens.")
        return
    if creating:
        if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
            st.error(f"Enter a password with {MIN_PASSWORD_LENGTH}–{MAX_PASSWORD_LENGTH} characters.")
            return
        if password != confirmation:
            st.error("The passwords do not match.")
            return

    try:
        if creating:
            if not create_account(username, password, demo_users):
                st.error("That username is unavailable. Choose another or sign in.")
                return
        elif not valid_sign_in(username, password, demo_users):
            st.error("The username or password is incorrect.")
            return
    except psycopg.Error as error:
        print(f"Grocery assistant account request failed ({type(error).__name__}).")
        st.error("Account service is temporarily unavailable. Try again shortly.")
        return

    st.session_state.user_id = username
    start_chat()
    st.rerun()


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
    setup_required = any(
        not isinstance(value := st.secrets.get(section, {}).get(key), str) or not value.strip()
        for section, key in REQUIRED_SETTINGS
    )
except StreamlitSecretNotFoundError:
    setup_required = True

if setup_required:
    st.info("App setup is required. Add the secrets shown in `.streamlit/secrets.toml.example`.")
    st.stop()

if "user_id" not in st.session_state:
    show_account_form()
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
            print(f"Grocery assistant request failed ({type(error).__name__}).", flush=True)
            st.session_state.turn_failed = True
            st.rerun()
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
