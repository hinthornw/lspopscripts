#!/usr/bin/env python3
import datetime
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

import langsmith
from tqdm import tqdm

project_name = "chat-langchain"
yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
client = langsmith.Client()


def download_data(
    project_name: str,
    nested: bool = False,
    since: datetime.datetime = yesterday,
    exclude_followups: bool = True,
    filename: str = "fetched_data.jsonl.gz",
):
    traces = client.list_runs(
        project_name=project_name, start_time=since, execution_order=1
    )
    batch_size = 10
    executor = ThreadPoolExecutor(max_workers=batch_size) if nested else None
    file_handle = (
        gzip.open(filename + ".gz", "wt", encoding="utf-8")
        if filename.endswith(".gz")
        else open(filename, "w")
    )
    try:
        if nested:
            pbar = tqdm()
            while True:
                batch = list(islice(traces, batch_size))
                if not batch:
                    break
                futures = [
                    executor.submit(client.read_run, run.id, load_child_runs=True)
                    for run in batch
                ]
                for future in as_completed(futures):
                    loaded_run = future.result()
                    file_handle.write(loaded_run.json() + "\n")
                pbar.update(len(batch))
        else:
            for run in tqdm(traces):
                if exclude_followups and run.inputs.get("chat_history"):
                    continue
                file_handle.write(run.json() + "\n")

    finally:
        if executor:
            executor.shutdown()
        file_handle.close()
    
    print(f"Saved to {filename}")


if __name__ == "__main__":
    import argparse

    from dateutil import parser as dateutil_parser

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=project_name)
    parser.add_argument("--nested", action="store_true")
    parser.add_argument("--since", default=yesterday.isoformat())
    parser.add_argument("--exclude-followups", action="store_true")
    parser.add_argument("--filename", default="fetched_data.jsonl.gz")
    args = parser.parse_args()
    download_data(
        project_name=args.project,
        nested=args.nested,
        since=dateutil_parser.parse(args.since),
        exclude_followups=args.exclude_followups,
        filename=args.filename,
    )
