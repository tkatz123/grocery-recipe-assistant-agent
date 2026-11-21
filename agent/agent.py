
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
import streamlit as st
from agent.agent_tools import search_recipes, get_recipe, search_prices, estimate_basket, get_profile, save_profile
from agent.prompts import SYSTEM_PROMPT, SUMMARY_PROMPT
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.store.postgres import PostgresStore
from dataclasses import dataclass

#============== Long Term Memory ==============

pool = ConnectionPool(
    st.secrets["neon"]["database_url"],
    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    check=ConnectionPool.check_connection,
    max_size=5,
    open=True,
)

store = PostgresStore(pool)

store.setup()

@dataclass
class Context:
    user_id: str

#============== Agent ==============

model = ChatOpenAI(
    model = 'gpt-4.1',
    api_key = st.secrets['openai']['api_key'],
    temperature = 0
)

summarization_model = ChatOpenAI(
    model = 'gpt-4.1-mini',
    api_key = st.secrets['openai']['api_key'],
    temperature = 0
)

convo_summarizer = SummarizationMiddleware(
    model = summarization_model,
    trigger = ("tokens", 40000),
    keep = ("tokens", 10000),
    summary_prompt = SUMMARY_PROMPT
)


checkpointer = InMemorySaver()

agent = create_agent(
    model = model,
    tools = [search_recipes, get_recipe, search_prices, estimate_basket, get_profile, save_profile],
    system_prompt = SYSTEM_PROMPT,
    checkpointer = checkpointer,
    middleware = [convo_summarizer],
        store = store,
    context_schema = Context
)

if __name__ == "__main__":

    result = agent.invoke(
        {"messages": [{'role': 'user', 'content': 'I have $50 dollars to spend at the store, what should I buy and what are 2 recipes I can make with those ingredients.'}]},
        {'configurable': {'thread_id': "id1"}}
    )

    for msg in result['messages']:
        print(msg.type)
        print(msg.content)

        if msg.type =="ai":
            print(msg.tool_calls)