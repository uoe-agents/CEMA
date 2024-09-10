import os
import openai
import logging
import argparse
import json
import itertools

from typing import List, Dict

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse scenario and temperature.")
    parser.add_argument('--scenario', type=int, default=1,
                        help='Scenario ID')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature value')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Whether to use test scenarios')
    args = parser.parse_args()
    return args


def get_system_prompts(system_config: Dict, scenario_config: Dict) -> List[str]:
    variables = system_config["variables"]
    variables_combinations = [dict(zip(variables.keys(), a)) for a in itertools.product(*variables.values())]

    ret = []
    for combination in variables_combinations:
        base_str = base_dict["base"]
        for var, val in combination.items():
            base_str = base_str.replace(f"<-{var}->", base_dict["variables"][var][val])
        base_str = base_str.replace("  ", " ")
        ret.append({"config": combination, "system": base_str})

    scenario_str = (f"Driving side: {scenario_config['regulations']['side']} traffic;\n"
                    f"Speed limit: {scenario_config['regulations']['speed_limit']} m/s;\n")

    return ret


def communicate_with_gpt4(system: str, prompt: str, **gpt_kwargs):
    client = openai.OpenAI(api_key=openai.api_key)
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        **gpt_kwargs
    )
    message = response.choices[0].message.content
    return message


if __name__ == '__main__':
    args = parse_arguments()
    scenario_id = args.scenario
    temperature = args.temperature

    base_path = os.path.join("configs", "base", ".json")
    base_dict = json.load(open(base_path, "r"))
    scenario_path = os.path.join("configs", f"{'test_' if args.test else ''}scenario_{scenario_id}", ".json")
    scenario_dict = json.load(open(scenario_path, "r"))
    system_prompt_combinations = get_system_prompts(base_dict, scenario_dict)

    response = communicate_with_gpt4(
        system_prompt, user_prompt, temperature=temperature)
    print(response)
