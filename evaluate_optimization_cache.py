import logging

from main import Coordinator

if __name__ == "__main__":
    logging.basicConfig(force=True, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    Coordinator().evaluate_opt_cache("experiments_results.json", opts_count=1)

