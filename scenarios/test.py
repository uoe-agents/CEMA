import json
import os
import gofi
import pickle
import argparse
import logging
from datetime import date
from matplotlib import pyplot as plt

import xavi
from util import setup_xavi_logging

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process scenario parameters.")
    parser.add_argument('--scenario', type=int, default=9, help='Scenario number')
    parser.add_argument('--query_idx', type=int, default=0, help='Query index')
    parser.add_argument('--n', type=int, default=30, help='Number of iterations')
    parser.add_argument('--load_existing', action="store_true",
                        default=False, help='Load existing data')
    parser.add_argument('--allow_hide_occluded', action="store_true",
                        default=False, help='Allow hiding occluded objects')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    scenario = args.scenario
    query_idx = args.query_idx
    n = args.n
    load_existing = args.load_existing
    allow_hide_occluded = args.allow_hide_occluded

    output_path = os.path.join("output", f"scenario_{scenario}")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    logger_path = os.path.join(output_path, "logs")
    if not os.path.exists(logger_path):
        os.makedirs(logger_path, exist_ok=True)
    setup_xavi_logging(log_dir=logger_path, log_name=f"scenario_{scenario}")

    logger.info(args)

    scenario_map = gofi.OMap.parse_from_opendrive(f"scenarios/maps/gofi-scenario{scenario - 8}.xodr")
    queries = json.load(open(f"scenarios/queries/query_scenario{scenario}.json", "r"))
    query = xavi.Query(**queries[query_idx])

    if scenario == 9:
        if not load_existing:
            oxavi_agent = pickle.load(open("output/scenario_9/agent_t100_mQueryType.WHY_NOT.pkl", "rb"))
            for mcts in oxavi_agent.cf_mcts.values():
                mcts._allow_hide_occluded = allow_hide_occluded
                mcts.n = n
            final_causes, efficient_causes = oxavi_agent.explain_actions(query)
        else:
            final_causes, efficient_causes = pickle.load(open(f"output/scenario_9/q_n{n}_t{query.t_query}_{query.type}.pkl", "rb"))

    elif scenario == 10:
        if not load_existing:
            oxavi_agent = pickle.load(open(f"output/scenario_10/agent_t{query.t_query}_mQueryType.WHY.pkl", "rb"))
            for mcts in oxavi_agent.cf_mcts.values():
                mcts._allow_hide_occluded = allow_hide_occluded
                mcts.n = n
            final_causes, efficient_causes = oxavi_agent.explain_actions(query)
        else:
            final_causes, efficient_causes = pickle.load(open("output/scenario_10/q_t120_mQueryType.WHY_NOT.pkl", "rb"))

    elif scenario == 11:
        if not load_existing:
            pass
        else:
            final_causes, efficient_causes = pickle.load(open("output/scenario_11/q_t140_mQueryType.WHY.pkl", "rb"))

    if not load_existing:
        file_path = os.path.join(output_path, f"q_n{n}_t{query.t_query}_m{query.type}.pkl")
        pickle.dump((final_causes, efficient_causes), open(file_path, "wb"))

    xavi.plot_explanation(final_causes, efficient_causes[0:2])
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"q_n{n}_t{query.t_query}_m{query.type}.png"))
    plt.show()
