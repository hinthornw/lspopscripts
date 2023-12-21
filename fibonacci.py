"""Generate a nested trace."""
import logging
from langsmith.run_helpers import traceable

logger = logging.getLogger(__name__)


@traceable()
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    logger.info("Running fibonacci with n=%s", args.n)
    fibonacci(args.n)
