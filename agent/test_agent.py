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

    # Usage: python -m agent.test_agent [scenario names ...] [--runs N]
    # Scores swing from run to run, so --runs 3 repeats every scenario and writes
    # <scenario>_run1.txt, _run2.txt, ... A scenario passes only if it passes every run.
    args = sys.argv[1:]
    runs = 1
    if "--runs" in args:
        position = args.index("--runs")
        runs = int(args[position + 1])
        del args[position:position + 2]
    if args:
        scenarios = [s for s in scenarios if s["name"] in args]

    for run, scenario in [(r, s) for r in range(1, runs + 1) for s in scenarios]:
        suffix = f"_run{run}" if runs > 1 else ""
        # A scenario is one "prompt", a list of "turns" (one conversation), or a list of
        # "conversations" (each a list of turns) by the same user, to test long-term memory.
        conversations = scenario.get("conversations") or [scenario.get("turns") or [scenario["prompt"]]]

        # A fresh test user per scenario per run, so saved profiles never leak between
        # scenarios or between runs. Deleted from the store once the scenario finishes.
        user_id = f"test_{results_dir.name}_{scenario['name']}{suffix}".replace(".", "_")  # namespace labels cannot contain periods
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
                # Label middleware messages (the budget check) so they read apart from the user.
                lines.append(f"--- {msg.type}" + (f" ({msg.name})" if getattr(msg, "name", None) else ""))
                lines.append(str(msg.content))
                if msg.type == "ai":
                    for call in msg.tool_calls:
                        lines.append(f"CALL {call['name']} {json.dumps(call['args'], ensure_ascii=False)}")

        store.delete(("users", user_id), "profile")

        out_path = results_dir / f"{scenario['name']}{suffix}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"done: {scenario['name']}{suffix}", flush=True)

    print(f"results in {results_dir}")