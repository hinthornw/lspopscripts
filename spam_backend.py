"""Just send a lot of data to the backend. Checking for run inconsistencies."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os

from langsmith import Client
import uuid

client = Client()
tm = datetime.utcnow()

NUM_RUNS = 1500
run_ids = [uuid.uuid4() for _ in range(NUM_RUNS)]


def create_llm_run(rid):
    proj = os.environ.get("LANGCHAIN_PROJECT", "test")
    client.create_run(
        id=rid,
        run_type="llm",
        inputs={"prompt": f"hello {proj} - {rid}"},
        outputs={},
        name="🦜",
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={"metadata": {"uid": str(rid)}},
    )


print(client)
# print("Most recent project name", next(client.list_projects()).name)
print("WRiting {} runs".format(NUM_RUNS))

with ThreadPoolExecutor() as executor:
    executor.map(
        create_llm_run,
        run_ids,
    )
print("Done inserting. Waiting")
import time

time.sleep(10)
batch_size = 50


def get_runs():
    all_runs = []
    for i in range(0, NUM_RUNS, batch_size):
        all_runs.extend(client.list_runs(run_ids=run_ids[i : i + batch_size]))
    return all_runs


results = {"run_ids": [str(id_) for id_ in run_ids], "missing": []}
with open(
    os.environ.get("LANGCHAIN_PROJECT", "test").replace(" ", "_") + ".json", "w"
) as f:
    import json

    json.dump(results, f)
for i in range(3):
    print(f"Attempt {i}")
    all_runs = get_runs()
    if len(all_runs) == NUM_RUNS:
        print("Success")
        break
    print(
        f"Only fetched {len(all_runs)} runs. Waiting 10 seconds and trying again. {NUM_RUNS - len(all_runs)} runs left to fetch"
    )
    time.sleep(10)
else:
    print("Failed to fetch all runs")
    missing = set(run_ids) - set(r.id for r in all_runs)
    results["missing"] = [str(id_) for id_ in list(missing)]
    with open(
        os.environ.get("LANGCHAIN_PROJECT", "test").replace(" ", "_") + ".json", "w"
    ) as f:
        import json

        json.dump(results, f)
    print(f"Missing runs:\n{missing}")
    exit(1)
