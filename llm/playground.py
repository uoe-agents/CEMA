""" Playground script for testing and debugging. """

import json
import igp2 as ip
import verbalize
import gpt


scenario_map = ip.Map.parse_from_opendrive("scenarios/maps/scenario1.xodr")
config = json.load(open("scenarios/configs/scenario1.json", encoding="utf-8"))

scenario_str = verbalize.scenario(config)
road_layout_str = verbalize.road_layout(scenario_map)

message = f"{scenario_str}\n\n{road_layout_str}\n\nDescribe the road in a short paragraph without referring to IDs."
print(len(message))

chat = gpt.Chat()
response = chat.prompt(message)
print(response)