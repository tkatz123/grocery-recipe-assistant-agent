import json
import sys
from datetime import datetime
from pathlib import Path
from agent.agent import agent, model, store, Context


if __name__ == "__main__":

    scenerios_path = Path(__file__).parent / "test_scenarios.json"

    with open(scenerios_path, encoding="utf-8") as f:

        scenarios = json.load(f)

    # One folder per run, named by time and model, all under test_results/.
    results_dir = Path("test_results") / f"{datetime.now():%Y-%m-%d_%H%M}_{model.model_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        scenarios = [s for s in scenarios if s["name"] == sys.argv[1]]

    for scenario in scenarios:
        # A scenario is one "prompt", a list of "turns" (one conversation), or a list of
        # "conversations" (each a list of turns) by the same user, to test long-term memory.
        conversations = scenario.get("conversations") or [scenario.get("turns") or [scenario["prompt"]]]

        # A fresh test user per scenario per run, so saved profiles never leak between
        # scenarios or between runs. Deleted from the store once the scenario finishes.
        user_id = f"test_{results_dir.name}_{scenario['name']}".replace(".", "_")  # namespace labels cannot contain periods
        context = Context(user_id=user_id)

        lines = []
        for number, turns in enumerate(conversations, start=1):
            # Each conversation gets its own thread_id: same user, fresh short-term memory.
            config = {"configurable": {"thread_id": f"{user_id}_conversation{number}"}}
            for turn in turns:
                result = agent.invoke({"messages": [{"role": "user", "content": turn}]},
                                      config=config, context=context)
            # The last result holds the whole conversation, loaded from the checkpointer.
            lines.append(f"=== conversation {number}")
            for msg in result["messages"]:
                lines.append(f"--- {msg.type}")
                lines.append(str(msg.content))
                if msg.type == "ai":
                    for call in msg.tool_calls:
                        lines.append(f"CALL {call['name']} {json.dumps(call['args'], ensure_ascii=False)}")

        store.delete(("users", user_id), "profile")

        out_path = results_dir / f"{scenario['name']}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"done: {scenario['name']}")

    print(f"results in {results_dir}")