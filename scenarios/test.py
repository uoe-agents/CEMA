import json

import igp2 as ip
import pickle
import json
import xavi
from util import setup_xavi_logging

setup_xavi_logging()
scenario_map = ip.Map.parse_from_opendrive("scenarios/maps/gofi-scenario1.xodr")
xavi_agent = pickle.load(open("output/scenario_9/agent_t100_mQueryType.WHY.pkl", "rb"))
queries = json.load(open("scenarios/queries/query_scenario9.json", "r"))
causes = xavi_agent.explain_actions(xavi.Query(**queries[0]))
