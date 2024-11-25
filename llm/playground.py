""" Playground script for testing and debugging. """

import json
import igp2 as ip
import xavi
import verbalize
import gpt


scenario_map = ip.Map.parse_from_opendrive("scenarios/maps/scenario1.xodr")
config = json.load(open("scenarios/configs/scenario0.json", encoding="utf-8"))
query = json.load(open("scenarios/queries/query_scenario0.json", encoding="utf-8"))
query = xavi.Query(**query[0])

print(verbalize.query(query, include_t_query=True))

# message = f"{scenario_str}\n\n{road_layout_str}\n\nDescribe the road in a short paragraph without referring to IDs."
# print(len(message))

# chat = gpt.Chat()
# response = chat.prompt(message)
# print(response)