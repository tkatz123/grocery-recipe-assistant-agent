
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware, after_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, RemoveMessage
import json
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

#============== Basket enforcement ==============
# Two failures the prompt alone did not prevent:
#   - run 5: the agent got within_budget false twice and answered with the over-budget list.
#   - November 24: after an over-budget result, the agent split the basket into one
#     estimate_basket call per recipe, then listed every item under one recipe's total,
#     so the total shown to the user was about $12 too low.
# This runs after every model call. If the model is about to give its final answer and the
# latest estimate_basket of this turn was over budget, or the answer does not state that
# call's estimated_total, the answer is discarded and the model is sent back with a reminder.
# At most one reminder per estimate_basket result: if the model then answers again without
# a new estimate (for example to say honestly that nothing fits), the answer goes through.

BUDGET_CHECK_NAME = "budget_check"  # the UI should not display messages with this name
MAX_BUDGET_CHECKS_PER_TURN = 3

BUDGET_REMINDER = (
    "Budget check: your last estimate_basket returned within_budget false, so this list is "
    "over the user's budget. Do not present it. Swap in cheaper products or choose a cheaper "
    "recipe, and call estimate_basket again with every product for all chosen recipes in ONE "
    "call. Only if no recipe in the results can fit the budget, say so plainly and show what "
    "the budget can actually cover."
)

TOTAL_REMINDER = (
    "Total check: your answer does not show the estimated_total from your last estimate_basket "
    "call (${total:.2f}). The total you show must come from ONE estimate_basket call that "
    "includes every product on the list for all the recipes you recommend. Call estimate_basket "
    "once with the complete list, then answer with exactly that call's items and total."
)


ERROR_REMINDER = (
    "Basket check: your last estimate_basket call returned an error, so you have no valid total. "
    "Read the error, fix the list, and call estimate_basket again with every product for all "
    "chosen recipes in ONE call. Never show a total you worked out yourself."
)

SPLIT_REMINDER = (
    "Basket check: your answer recommends {missing}, which your last estimate_basket call did "
    "not include. Do not total recipes in separate calls. Call estimate_basket ONCE with every "
    "product for all the recipes you recommend, check within_budget, and answer with exactly "
    "that call's items and total."
)


def _mentions_total(text, total):
    """True if the answer states this dollar total (as 32.66, or 20 for a whole-dollar total)."""
    forms = {f"{total:.2f}"}
    if float(total).is_integer():
        forms.add(f"${int(total)}")
    return any(form in text for form in forms)


@after_model(can_jump_to=["model"])
def enforce_budget(state, runtime):
    messages = state["messages"]
    last = messages[-1]
    # Only a final answer (no tool calls) is checked.
    if not isinstance(last, AIMessage) or last.tool_calls:
        return None

    checks_this_turn = 0
    reminded_since_estimate = False
    newest = None            # the latest estimate_basket result of this turn, error or not
    estimate = None          # the latest successful one
    earlier_recipes = set()  # recipe names in earlier successful estimates this turn
    # Walk back through this turn only, newest first, stopping at the user's message.
    for msg in reversed(messages[:-1]):
        if isinstance(msg, HumanMessage):
            if msg.name == BUDGET_CHECK_NAME:
                checks_this_turn += 1
                if newest is None:
                    reminded_since_estimate = True
                continue
            break
        if isinstance(msg, ToolMessage) and msg.name == "estimate_basket":
            try:
                result = json.loads(msg.content)
            except (TypeError, ValueError):
                continue
            if not isinstance(result, dict):
                continue
            if newest is None:
                newest = result
            if "estimated_total" in result:
                if estimate is None:
                    estimate = result
                else:
                    # "uncovered" is keyed by recipe name for every recipe in that call.
                    earlier_recipes |= set(result.get("uncovered", {}))

    if newest is None or reminded_since_estimate or checks_this_turn >= MAX_BUDGET_CHECKS_PER_TURN:
        return None

    answer = str(last.content)
    reminder = None
    if "estimated_total" not in newest:
        reminder = ERROR_REMINDER
    elif estimate.get("within_budget") is False:
        reminder = BUDGET_REMINDER
    elif not _mentions_total(answer, estimate["estimated_total"]):
        reminder = TOTAL_REMINDER.format(total=estimate["estimated_total"])
    else:
        # A recipe totaled in an earlier call but not the last one, yet named in the answer,
        # means the basket was split across calls (seen November 24).
        final_recipes = set(estimate.get("uncovered", {}))
        missing = [name for name in earlier_recipes - final_recipes
                   if name.casefold() in answer.casefold()]
        if missing:
            reminder = SPLIT_REMINDER.format(missing=", ".join(sorted(missing)))

    if reminder:
        return {
            "messages": [RemoveMessage(id=last.id),
                         HumanMessage(content=reminder, name=BUDGET_CHECK_NAME)],
            "jump_to": "model",
        }
    return None


agent = create_agent(
    model = model,
    tools = [search_recipes, get_recipe, search_prices, estimate_basket, get_profile, save_profile],
    system_prompt = SYSTEM_PROMPT,
    checkpointer = checkpointer,
    middleware = [convo_summarizer, enforce_budget],
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