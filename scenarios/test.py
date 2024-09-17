import json
import os
import gofi
import pickle
import argparse
import logging
from matplotlib import pyplot as plt

import xavi
import oxavi
import igp2 as ip
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
    setup_xavi_logging(log_dir=logger_path, log_name=f"test_s{scenario}")

    logger.info(args)

    scenario_map = gofi.OMap.parse_from_opendrive(f"scenarios/maps/scenario{scenario}.xodr")
    config = json.load(open(f"scenarios/configs/scenario{scenario}.json", "r"))
    queries = json.load(open(f"scenarios/queries/query_scenario{scenario}.json", "r"))
    query = xavi.Query(**queries[query_idx])

    if scenario == 1:
        if not load_existing:
            xavi_agent = pickle.load(open(f"output/scenario_1/agent_t{query.t_query}_m{query.type}.pkl", "rb"))
            for mcts in xavi_agent.cf_mcts.values():
                mcts._allow_hide_occluded = allow_hide_occluded
                mcts.n = n
            final_causes, efficient_causes = xavi_agent.explain_actions(query)
    elif scenario == 9:
        if not load_existing:
            oxavi_agent = pickle.load(open("output/scenario_9/agent_t100_mQueryType.WHY_NOT.pkl", "rb"))
            for mcts in oxavi_agent.cf_mcts.values():
                mcts._allow_hide_occluded = allow_hide_occluded
                mcts.n = n
            final_causes, efficient_causes = oxavi_agent.explain_actions(query)

    elif scenario == 10:
        if not load_existing:
            oxavi_agent = pickle.load(open(f"output/scenario_10/agent_t{query.t_query}_mQueryType.WHY.pkl", "rb"))
            oxavi.OXAVITree.STOP_CHANCE = 1.0
            oxavi.OFollowLaneCL.IGNORE_VEHICLE_IN_FRONT_CHANCE = 0.0
            for mcts in oxavi_agent.cf_mcts.values():
                mcts._allow_hide_occluded = allow_hide_occluded
                mcts.n = n
                mcts.reward = ip.Reward(factors=config["agents"][0]["mcts"]["reward_factors"])
            final_causes, efficient_causes = oxavi_agent.explain_actions(query)

    elif scenario == 11:
        if not load_existing:
            oxavi_agent = pickle.load(open(f"output/scenario_11/agent_t100_mQueryType.WHY_NOT.pkl", "rb"))
            # oxavi.OXAVITree.STOP_CHANCE = 1.0
            # oxavi.OFollowLaneCL.IGNORE_VEHICLE_IN_FRONT_CHANCE = 0.0
            for mcts in oxavi_agent.cf_mcts.values():
                mcts._allow_hide_occluded = allow_hide_occluded
                mcts.n = n
                mcts.reward = ip.Reward(factors=config["agents"][0]["mcts"]["reward_factors"])
            final_causes, efficient_causes = oxavi_agent.explain_actions(query)

    if not load_existing:
        file_path = os.path.join(output_path, f"q_n{n}_t{query.t_query}_m{query.type}.pkl")
        pickle.dump((final_causes, efficient_causes), open(file_path, "wb"))
    else:
        final_causes, efficient_causes = pickle.load(
            open(f"output/scenario_{scenario}/q_n{n}_t{query.t_query}_m{query.type}.pkl", "rb"))

    xavi.plot_explanation(final_causes, efficient_causes[0:2], query, uniform_teleological=False)

    # import pandas as pd
    # from xavi.util import get_coefficient_significance
    # from sklearn.linear_model import LogisticRegression
    #
    # for i in [2, 3]:
    #     x, y, _  = efficient_causes[i]
    #     y = pd.Series(y)
    #     occluded = x["2_occluded"] == 1
    #     xo, yo = x[occluded], y[occluded]
    #     xno, yno = x[~occluded], y[~occluded]
    #     model_o = LogisticRegression().fit(xo, yo)
    #     coefs_o = get_coefficient_significance(xo, yo, model_o)
    #     model_no = LogisticRegression().fit(xno, yno)
    #     coefs_no = get_coefficient_significance(xno, yno, model_no)
    #     xavi.plot_explanation(final_causes, (coefs_o, coefs_no), query)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"q_n{n}_t{query.t_query}_m{query.type}.png"))
    plt.show()
