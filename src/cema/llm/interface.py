""" This module provides the interface to the various LMs. """
import logging
import pickle
from typing import Union, List, Dict

from vllm import LLM, SamplingParams

from cema.llm.chat import ChatHandler


logger = logging.getLogger(__name__)


class LMInterface():
    """ Common interface for interacting with LLMs. """

    def __init__(self,
                 model: Union[str, LLM],
                 chat_handler: ChatHandler,
                 sampling_params: SamplingParams = None):
        """ Initialize the LMInterface.

        Args:
            model: The model to use for the interface.
            chat_handler: The chat handler to use for the interface.
            sampling_params: The sampling parameters to use for the interface.
        """
        self._model = self.load_model(model)
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.5)
        self._sampling_params = sampling_params
        self._chat_handler = chat_handler

        self._causes = None
        self._agent = None

    def load_model(self, model: Union[str, LLM]) -> LLM:
        """ Load a new model for the interface."""
        if isinstance(model, str):
            return LLM(model=model)
        return model

    def load_scenario(self, scenario_id: int, scenario_path = None):
        """ Load a driving scenario with the given ID into memory.
        This function requires both the agent and the causes to be loaded.
        If you have not run the explanation generation for this scenario yet,
        then you should use the following command to obtain the pickled agent and causes:

        uv run cema explain SCENARIO_ID

        Args:
            scenario: The ID of the scenario to load.
            scenario_path: The optional path to directory of scenario files.
        """
        # if isinstance(scenario: str):
            # self._causes = pickle.load(open(f"scenarios/llm/scenario_{scenario}.pkl", "rb"))
        # return json.load(open(f"scenarios/llm/scenario_{scenario}.json", "r", encoding="utf-8"))
