import json
import igp2 as ip
import pickle

from matplotlib import pyplot as plt

import xavi
from util import setup_xavi_logging

setup_xavi_logging()
scenario_map = ip.Map.parse_from_opendrive("scenarios/maps/gofi-scenario1.xodr")
# xavi_agent = pickle.load(open("output/scenario_9/agent_t100_mQueryType.WHY.pkl", "rb"))
queries = json.load(open("scenarios/queries/query_scenario9.json", "r"))
# final_causes, efficient_causes = xavi_agent.explain_actions(xavi.Query(**queries[0]))
final_causes, efficient_causes = pickle.load(open("output/scenario_9/q_t100_mQueryType.WHY.pkl", "rb"))

xavi.plot_explanation(final_causes, efficient_causes[0:2])
plt.show()
