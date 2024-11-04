import logging
import os.path
import pickle
import sys
from typing import List

import igp2 as ip
from igp2.core.config import Configuration
import xavi
import oxavi
import gofi
import argparse
import tqdm
import matplotlib.pyplot as plt

from evaluation import load_scenario
from plotting import plot_sampling_results

logger = logging.Logger(__name__)



def main(args):
    agent, query = load_scenario(args.sid, args.qid)
    query_str = f"n30_t{query.t_query}_m{query.type}"

    sampling_size_results = {}
    for sample_size in (pbar := tqdm.tqdm([100, 250, 500, 750, 1000])):
        results = []
        logger.info("Generating explanations for sample size %d", sample_size)
        agent._cf_n_samples = sample_size
        causes = agent.explain_actions(query)
        results.append(causes)
        sampling_size_results[sample_size] = results

    plot_sampling_results(sampling_size_results, "output/test", query_str)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process scenario parameters.")
    parser.add_argument('--sid', type=int, default=1, help='Scenario ID')
    parser.add_argument('--qid', type=int, default=0, help='Index of query to run')
    arguments = parser.parse_args()

    sys.exit(main(arguments))
