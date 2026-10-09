"""Reproduce answer-buffer rebinding without making model requests.

Run in the same RLM installation as the experiment:
    sources/longcot/.venv/bin/python /path/to/check-answer-buffer.py

This checks the environment protocol only; it does not rerun benchmark tasks.
"""
import json

from rlm.environments.ipython_repl import IPythonREPL

results = []
for rebind in (False, True):
    env = IPythonREPL(kernel_mode="subprocess", cell_timeout=10, startup_timeout=20)
    try:
        if rebind:
            env.execute_code("answer = {'q1': 'Bool'}")
        result = env.execute_code('answer["content"] = "CHECK"\nanswer["ready"] = True\nprint(answer)')
        results.append({"rebind_answer": rebind, "stdout": result.stdout.strip(),
                        "stderr": result.stderr, "final_answer": result.final_answer})
    finally:
        env.cleanup()
assert results[0]["final_answer"] == "CHECK", results
assert results[1]["final_answer"] is None and not results[1]["stderr"], results
print(json.dumps(results, ensure_ascii=False, indent=2))
