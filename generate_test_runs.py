"""Populate a dataset with test runs containing random data."""
import random
import math
from langsmith import Client
from langsmith.evaluation.evaluator import run_evaluator
from langchain.smith.evaluation.name_generation import random_name
from langchain.smith import RunEvalConfig
from langsmith import traceable
from concurrent.futures import ThreadPoolExecutor


import subprocess
from functools import cache
import threading
from typing import Optional

_git_lock = threading.RLock()


def run_git_command(command):
    with _git_lock:  # Ensure thread safety for git commands
        try:
            return subprocess.check_output(
                ["git"] + command, encoding="utf-8", stderr=subprocess.DEVNULL
            ).strip()
        except subprocess.CalledProcessError:
            return None


@cache
def convert_to_https_url(url: str) -> Optional[str]:
    if url.startswith("git@"):
        url = url.replace("git@", "https://").replace(":", "/")
    elif url.startswith("http://"):
        url = url.replace("http://", "https://")
    return url if url.startswith("https://") else None


@cache
def get_git_info():
    remote_url = run_git_command(["config", "--get", "remote.origin.url"])
    https_url = convert_to_https_url(remote_url) if remote_url else None

    git_info = {
        "commit": run_git_command(["rev-parse", "HEAD"]),
        "branch": run_git_command(["rev-parse", "--abbrev-ref", "HEAD"]),
        "tag": run_git_command(["describe", "--tags", "--exact-match"]),
        "dirty": run_git_command(["status", "--porcelain"]) != "",
        "author_name": run_git_command(["log", "-1", "--format=%an"]),
        "author_email": run_git_command(["log", "-1", "--format=%ae"]),
        "commit_message": run_git_command(["log", "-1", "--format=%B"]),
        "commit_time": run_git_command(["log", "-1", "--format=%cd"]),
        "remote_url": https_url,
    }

    return git_info


@run_evaluator
def random_evaluator(run, example=None):
    return {"key": "my_random", "score": random.random() * 10}


@run_evaluator
def randomly_null(run, example=None):
    score = random.random()
    if score > 0.5:
        score = None
    return {"key": "randomly_null", "score": score}


@run_evaluator
def big_numbers(run, example=None):
    # Score is exponentially distibuted with a mean of 20
    return {"key": "big_numbers", "score": random.expovariate(1 / 20)}


@run_evaluator
def log_loss(run, example=None):
    # Note, log loss can go negative
    return {"key": "log_loss", "score": math.log(random.random())}


@run_evaluator
def hundred_binary(run, example=None):
    return {"key": "hundred_binary", "score": 100 if random.random() > 0.5 else 0}


@run_evaluator
def always_one(run, example=None):
    return {"key": "always_1", "score": 1}


@run_evaluator
def feedback_stats(run, example=None):
    # Just to check coercion
    return {"key": "feedback_stats_", "score": 1}


_DATASET_NAME = "Random KV Dataset"


def create_dataset():
    client = Client()
    ds = client.create_dataset(_DATASET_NAME, description="A dataset for testing")

    examples = [
        dict(
            inputs=dict(
                question=f"What is the meaning of life on on planet {random_name()}?"
            ),
            outputs=dict(answer=str(random.random())),
        )
        for _ in range(100)
    ]
    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=ds.id,
    )


@traceable()
def my_model(question):
    if random.random() < 0.05:
        raise ValueError("Random error")
    return {"answer": str(random.random())}


def run_evaluations():
    client = Client()
    print(client)
    client.run_on_dataset(
        dataset_name=_DATASET_NAME,
        llm_or_chain_factory=my_model,
        evaluation=RunEvalConfig(
            custom_evaluators=[
                random_evaluator,
                randomly_null,
                big_numbers,
                log_loss,
                always_one,
                hundred_binary,
                feedback_stats,
            ],
        ),
        verbose=True,
        project_metadata={
            "random_number": random.random(),
            **get_git_info(),
            "a nested val": {"a": {"b": {"c": 1}}},
            "🦜": ["a", "list", "of", "things"],
            "disorderly": 5 if random.random() < 0.5 else {"id": "foo"},
        },
    )


def main(n: int = 1):
    client = Client()
    print(client)
    if not client.has_dataset(dataset_name=_DATASET_NAME):
        print("Creating dataset")
        create_dataset()
    print("Running evaluations")
    if n == 1:
        run_evaluations()
    else:
        with ThreadPoolExecutor(max_workers=10) as executor:
            for _ in range(n):
                executor.submit(run_evaluations)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()
    main(args.n)
