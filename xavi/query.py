from dataclasses import dataclass


@dataclass
class Query:
    """ Dataclass to store parsed query information. """
    type: str
    time: int
    action: str = None
    agent_id: int = None
    negative: bool = None
