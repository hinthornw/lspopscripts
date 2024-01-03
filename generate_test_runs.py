"""Populate a dataset with test runs containing random data."""
import random
import math
from langsmith import Client
from langsmith.evaluation.evaluator import run_evaluator
from langchain.smith.evaluation.name_generation import random_name
from langchain.smith import RunEvalConfig
from langsmith import traceable
from concurrent.futures import ThreadPoolExecutor


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
        tags=["random", "test"],
        project_metadata={
            "random_number": random.random(),
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
