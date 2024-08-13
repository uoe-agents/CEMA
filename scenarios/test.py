import json
import os
import sys
import gofi
import pickle

from matplotlib import pyplot as plt

import xavi
from util import setup_xavi_logging

setup_xavi_logging()

scenario = 11
query_idx = 0
output_path = os.path.join("output", f"scenario_{scenario}")

scenario_map = gofi.OMap.parse_from_opendrive(f"scenarios/maps/gofi-scenario{scenario - 8}.xodr")
queries = json.load(open(f"scenarios/queries/query_scenario{scenario}.json", "r"))
query = xavi.Query(**queries[query_idx])

if scenario == 9:
    # oxavi_agent = pickle.load(open("output/scenario_9/agent_t100_mQueryType.WHY.pkl", "rb"))
    # final_causes, efficient_causes = oxavi_agent.explain_actions(query)
    final_causes, efficient_causes = pickle.load(open("output/scenario_9/q_t100_mQueryType.WHY.pkl", "rb"))

elif scenario == 10:
    # oxavi_agent = pickle.load(open("output/scenario_10/agent_t120_mQueryType.WHY_NOT.pkl", "rb"))
    # for mcts in oxavi_agent.cf_mcts.values(): mcts._allow_hide_occluded = True
    # final_causes, efficient_causes = oxavi_agent.explain_actions(query)
    final_causes, efficient_causes = pickle.load(open("output/scenario_10/q_t120_mQueryType.WHY_NOT.pkl", "rb"))

elif scenario == 11:
    final_causes, efficient_causes = pickle.load(open("output/scenario_11/q_t140_mQueryType.WHY.pkl", "rb"))

else:
    sys.exit(-1)

# file_path = os.path.join(output_path, f"q_t{query.t_query}_m{query.type}.pkl")
# pickle.dump((final_causes, efficient_causes), open(file_path, "wb"))

xavi.plot_explanation(final_causes, efficient_causes[0:2])
plt.tight_layout()
plt.savefig(os.path.join(output_path, f"q_t{query.t_query}_m{query.type}.png"))
plt.show()
