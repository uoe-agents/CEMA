import os
import json
from dotmap import DotMap


def load_query(args):
    if "scenario" in args:
        path = os.path.join("scenarios", "queries", f"query_scenario{args.scenario}.json")
    elif "query_path" in args:
        path = args.query_path
    else:
        raise ValueError("No query was specified!")
    return json.load(open(path, "r"), object_hook=lambda d: DotMap(**d))


def query_parser(args):
    query_info = {"query_type": None, "maneuver": None, "time": None, "aid": None, "negative": None}
    query = load_query(args)
    if args.query_type == 1:
        # why query
        query_info["query_type"] = "why"
        query_info["maneuver"] = query.why.maneuver
        query_info["time"] = query.why.time
    elif args.query_type == 2:
        # why not query
        query_info["query_type"] = "whynot"
        query_info["maneuver"] = query.whynot.maneuver
        query_info["time"] = query.whynot.time
    elif args.query_type == 3:
        # what if query
        query_info["query_type"] = "whatif"
        query_info["aid"] = query.whatif.agent_id
        query_info["maneuver"] = query.whatif.maneuver
        query_info["negative"] = query.whatif.negative
        query_info["time"] = query.whatif.time
    elif args.query_type == 4:
        # what if query
        query_info["query_type"] = "what"
        query_info["time"] = query.what.time
    else:
        raise TypeError('Type is not allowed')

    return query_info
