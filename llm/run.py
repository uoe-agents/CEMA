""" 
Module containing the main function for running evaluation with GPT-4o. 

Notes: 
1. The functioning of this module requires having previously run and saved the agent in a scenario.
2. If CEMA is used with GPT-4o, then the CEMA causes must have also already been generated.
"""

import os
import sys
import logging
import argparse
import json
import itertools
from typing import List, Dict

import igp2 as ip
from llm import verbalize
from llm import gpt
import scenarios.util as sutil
from scenarios.evaluation import load_scenario

logger = logging.getLogger(__name__)



# def get_system_prompts(system_config: Dict, scenario_config: Dict) -> List[str]:
#     variables = system_config["variables"]
#     variables_combinations = [dict(zip(variables.keys(), a)) for a in itertools.product(*variables.values())]

#     ret = []
#     for combination in variables_combinations:
#         base_str = base_dict["base"]
#         for var, val in combination.items():
#             base_str = base_str.replace(f"<-{var}->", base_dict["variables"][var][val])
#         base_str = base_str.replace("  ", " ")
#         ret.append({"config": combination, "system": base_str})

#     scenario_str = (f"Driving side: {scenario_config['regulations']['side']} traffic;\n"
#                     f"Speed limit: {scenario_config['regulations']['speed_limit']} m/s;\n")

#     return ret

SYSTEM_PROMPT = """You are the explanation module of an autonomous driving system.
You explain the actions of the autonomous driving system in various scenarios when prompted by a user query.
Reply with helpful and detailed answers which include driving action justification."""


def main(args) -> int:
    """ Verbalize the scenario and prompt GPT-4o for an explanation. """

    # Create folder structure
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", f"scenario_{args.scenario}")
    os.makedirs(output_path, exist_ok=True)
    log_path = os.path.join(output_path, "logs")
    os.makedirs(log_path, exist_ok=True)


    # Setup logging
    sutil.setup_xavi_logging(log_dir=log_path, log_name="llm")
    logging.getLogger("xavi.explainer").setLevel(logging.WARNING)
    logger.info("Current arguments: %s", args)


    # Load scenario and query and observations
    config = sutil.load_config(args)
    agent, query = load_scenario(args.scenario, args.query)


    # Verbalize the road layout for the scenario
    scenario = verbalize.scenario(config, agent.scenario_map, agent.observations)


    # Create Chat object
    chat = gpt.Chat(system_prompt=SYSTEM_PROMPT)
    chat.prompt(scenario)


    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse scenario and temperature.")
    parser.add_argument('--scenario', type=int, default=None,
                        help='Scenario ID.. If not provided, the test scenario will be used.')
    parser.add_argument("--query", type=int, default=0,
                        help="Index of query to generate explanations for.")
    cmd_args = parser.parse_args()

    sys.exit(main(cmd_args))
