""" Command line interface for generating and evaluations explanation with CEMA. """
import os
import sys
import random
import logging
import pickle
from typing_extensions import Annotated

import typer
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm.contrib.logging import logging_redirect_tqdm

import igp2 as ip
import gofi
from cema.scripts.util import generate_random_frame, load_config, parse_query, \
    create_agent, run_simple_simulation
from cema.scripts.evaluation import sampling_robustness, distribution_robustness, load_scenario
from cema.scripts.plotting import plot_distribution_results, plot_sampling_results, plot_explanation
from cema import xavi, oxavi, setup_cema_logging


logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def explain(
    ctx: typer.Context,
    scenario: Annotated[
        int,
        typer.Argument(help="The ID of the scenario to execute.", metavar="S", min=0)
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed of the simulation.")] = 21,
    fps: Annotated[int, typer.Option(help="Framerate of the simulation.", min=5, max=100)] = 20,
    config_path: Annotated[str, typer.Option(help="Path to a scenario configuration file.")] = None,
    query_path: Annotated[str, typer.Option(help="Path to load a query.")] = None,
    save_causes: Annotated[
        bool,
        typer.Option(help="Whether to pickle the causes for each query.")
    ] = False,
    save_agent: Annotated[
        bool,
        typer.Option(help="Whether to pickle the agent for each query.")
    ] = False,
    plot: Annotated[bool, typer.Option(help="Whether to display plots of the simulation.")] = False,
    sim_only: Annotated[bool, typer.Option(help="If true then do not execute queries.")] = False,
    debug: Annotated[bool, typer.Option(help="Whether to display debugging plots.")] = False,
    carla: Annotated[
        bool,
        typer.Option(help="Whether to use CARLA as the simulator instead of the simple simulator.")
    ] = False
):
    """ Explain a scenario with the given ID from a config file. """

    if not os.path.exists("output"):
        os.mkdir("output")
    output_path = os.path.join("output", f"scenario_{scenario}")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    setup_cema_logging(log_dir=os.path.join(output_path, "logs"), log_name="run")

    logger.info(ctx.args)

    config = load_config(config_path, scenario)
    queries = parse_query(query_path, scenario)

    scenario_map = gofi.OMap.parse_from_opendrive(config["scenario"]["map_path"])

    # Get run parameters
    seed = config["scenario"]["seed"] if "seed" in config["scenario"] else seed

    random.seed(seed)
    np.random.seed(seed)
    np.seterr(divide="ignore")

    ip_config = ip.core.config.Configuration()
    ip_config.set_properties(**config["scenario"])
    oxavi.OFollowLaneCL.IGNORE_VEHICLE_IN_FRONT_CHANCE = \
        config["scenario"].get("ignore_vehicle_in_front_chance", 0.0)

    frame = generate_random_frame(scenario_map, config)

    fps = config["scenario"]["fps"] if "fps" in config["scenario"] else fps
    try:
        simulation = gofi.OSimulation(scenario_map, fps)

        for agent_config in config["agents"]:
            agent, rolename = create_agent(agent_config, scenario_map, frame, fps, carla)
            simulation.add_agent(agent, rolename=rolename)

        if plot:
            xavi.plot_simulation(simulation, debug=debug)
            plt.show()
        result = run_simple_simulation(
            simulation, plot, sim_only, queries, config, output_path, save_causes, save_agent)
    except Exception as e:
        logger.exception(msg=str(e), exc_info=e)
        result = False
    finally:
        del simulation
    return result


@app.command()
def llm(
    model : Annotated[
        str,
        typer.Argument(help="LLM model name.", metavar="M")
    ] = "meta-llama/Llama-3.2-1B-Instruct",
    scenario: Annotated[
        int,
        typer.Argument(help="The ID of the scenario to execute.", metavar="S", min=0)
    ] = None
):
    """ Explain a scenario with the given ID and configuration using an LLM model. """
    print(model, scenario)


@app.command()
def evaluate(
    scenario: Annotated[
        int,
        typer.Argument(help="The ID of the scenario to execute.", metavar="S", min=0)
    ] = None,
    query: Annotated[
        int,
        typer.Argument(help="The index of the query to execute.", metavar="Q", min=0)
    ] = None,
    norobust: Annotated[
        bool,
        typer.Option(help="Do not run robustness evaluations.")
    ] = False
):
    """ Evaluate the robustness of the explanation generation with increasing sample sizes
    and distribution smoothing. Also plot explanation reults."""

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # Setup output directories
    output_path = os.path.join("output", f"scenario_{scenario}")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    logger_path = os.path.join(output_path, "logs")
    if not os.path.exists(logger_path):
        os.makedirs(logger_path, exist_ok=True)
    plot_path = os.path.join(output_path, "plots")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)


    # Setup logging
    setup_cema_logging(log_dir=logger_path, log_name="evaluation")
    logging.getLogger("xavi.explainer").setLevel(logging.WARNING)
    logging.getLogger("oxavi.oexplainer").setLevel(logging.WARNING)


    # Load scenario and query
    logger.info("Loading scenario and query . . .")
    agent, query = load_scenario(scenario, query)
    query_str = f"t{query.t_query}_m{query.type}"

    plot_path_query = os.path.join(plot_path, query_str)
    if not os.path.exists(plot_path_query):
        os.makedirs(plot_path_query, exist_ok=True)


    # Plot causal attributions
    logger.info("Generating plots of explanation for query . . .")
    causes = pickle.load(open(os.path.join(output_path, f"q_{query_str}.pkl"), "rb"))
    if query.type == xavi.QueryType.WHAT_IF:
        cf_action_group = causes[0]
        logger.info(cf_action_group)
        final_causes = causes[1]
        efficient_causes = causes[2]
    else:
        final_causes = causes[0]
        efficient_causes = causes[1]
    plot_explanation(final_causes, efficient_causes[0:2], query_str, plot_path_query)


    if norobust:
        return 0


    # Run explanation generation with increasing uniformity
    logger.info("Running alpha smoothing robustness evaluation . . .")
    distribution_path = os.path.join(output_path, f"distribution_{query_str}.pkl")
    if not os.path.exists(distribution_path):
        with logging_redirect_tqdm():
            distribution_results = distribution_robustness(10, agent, query)
        pickle.dump(distribution_results, open(distribution_path, "wb"))
    else:
        distribution_results = pickle.load(open(distribution_path, "rb"))
    plot_distribution_results(distribution_results, plot_path_query, query_str)


    # Run explanation generation with increasing sample sizes
    logger.info("Running sample size robustness evaluation . . .")
    sampling_path = os.path.join(output_path, f"sampling_{query_str}.pkl")
    if not os.path.exists(sampling_path):
        with logging_redirect_tqdm():
            sampling_results = sampling_robustness(10, agent, query)
        pickle.dump(sampling_results, open(sampling_path, "wb"))
    else:
        sampling_results = pickle.load(open(sampling_path, "rb"))
    plot_sampling_results(sampling_results, plot_path_query, query_str)

    return 0


def cli():
    """ Run the command line interface. """
    app()


if __name__ == "__main__":
    sys.exit(cli())
