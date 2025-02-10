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

from llm import verbalize
from llm import gpt
import scenarios.util as sutil
from scenarios.evaluation import load_scenario

logger = logging.getLogger(__name__)



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
    verbalized_scenario = verbalize.scenario(
        config,
        agent.scenario_map,
        agent.observations,
        query,
        add_road_layout=False,
        f_subsample=2,
        control_signals=["position", "speed"])
    logger.info("Verbalized scenario: %s", verbalized_scenario)

    # Create Chat object
    chat = gpt.Chat()
    chat.prompt(verbalized_scenario)

    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse scenario and temperature.")
    parser.add_argument('--scenario', type=int, default=None,
                        help='Scenario ID.. If not provided, the test scenario will be used.')
    parser.add_argument("--query", type=int, default=0,
                        help="Index of query to generate explanations for.")
    cmd_args = parser.parse_args()

    sys.exit(main(cmd_args))
