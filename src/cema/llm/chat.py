""" This module is responsible for managing the chat interactions with the various models. """
import logging
import json
from itertools import chain
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
SYSTEM_PROMPT = "You are the explanation module of an autonomous driving system."


class ChatHandlerConfig:
    """ Configuration class for the ChatHandler. """

    def __init__(self, config: Dict[str, Any]):
        """ Initialize the ChatHandlerConfig.

        Args:
            config: The configuration dictionary.
        """
        self._config = config

    @property
    def model(self) -> str:
        """ Return the model name. """
        return self._config["model"]

    @property
    def system_prompt(self) -> str:
        """ Return the system prompt. """
        return self._config["system_prompt"]

    @property
    def auto_log(self) -> bool:
        """ Return whether to automatically log the chat history. """
        return self._config["auto_log"]


class ChatHandler:
    """ Base class for handling chat interactions with various model. """

    def __init__(self,
                 model,
                 system_prompt: str,
                 auto_log: bool = True):
        """ Initialize the ChatHandler.

        Args:
            model: The model to use for the chat handler.
            system_prompt: The system prompt to use for the chat handler.
            auto_log: Whether to automatically log the chat history.
        """
        self._model = model

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        self._system_prompt = system_prompt
        if auto_log:
            logger.warning("[SYSTEM] %s", system_prompt)
        self._auto_log = auto_log

        self._user_history = []
        self._response_history = []

    def interact(self, prompt: str, use_history: bool = True, **kwargs) -> str:
        """ Ask the model to generate a response to the given prompt. May be
        overriden to provide additional functionality.

        Args:
            prompt: The prompt to respond to.
            use_history: Whether to use the chat history as context.
            **kwargs: Additional keyword arguments for the model.
        """
        # Setup message history if needed
        messages = [
            {"role": "system", "content": self.system_prompt},
        ] + self.history if use_history else []

        # Create user message
        user_message = {"role": "user", "content": prompt}
        messages.append(user_message)
        self._user_history.append(user_message)

        if self._auto_log:
            logger.warning("\n[USER] %s", prompt)

        self.model.chat(messages, )

    def reset_history(self):
        """ Reset all of the chat history. """
        self._user_history = []
        self._response_history = []

    def save_chat_history(self, path: str):
        """ Save the chat history to a file. """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f)

    def save_response_history(self, path: str):
        """ Save the chat history to a file. """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.response_history, f)

    @property
    def model(self):
        """ Return the model/client used by the chat handler. """
        return self._model

    @property
    def history(self) -> Dict[str, List[Dict[str, str]]]:
        """ Return the combined user and assisstant chat history as alternating messages. """
        assistant_history = [{"role": "assistant", "content": r.choices[0].message.content}
                             for r in self.response_history]
        return list(chain.from_iterable(zip(self._user_history, assistant_history)))

    @property
    def user_history(self) -> List[Dict[str, str]]:
        """ Return the message history of the user. """
        return self._user_history

    @property
    def response_history(self) -> List[Dict[str, str]]:
        """ Return the full response history of the assistant. """
        return self._response_history

    @property
    def system_prompt(self) -> str:
        """ Return the system prompt. """
        return self._system_prompt


class ChatHandlerFactory:
    """ Factory class for creating ChatHandlers. """
    HANDLERS = {
        "llama": ChatHandler
    }

    @classmethod
    def create_chat_handler(cls, config: ChatHandlerConfig) -> ChatHandler:
        """ Create a new ChatHandler based on the given parameters.

        Args:
            config: The configuration for the chat handler.
        """
        assert config.model in cls.HANDLERS, f"Model {config.model} not supported."
        return cls(config.model, config.system_prompt, config.auto_log)
