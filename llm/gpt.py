""" Helper classes and functions for interacting with the OpenAI API over multiple turns. """

import logging
import json
from itertools import chain
from typing import List, Dict
import openai


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the explanation module of an autonomous driving system.
You explain the actions of the autonomous driving system in various scenarios when prompted by a user query.
Reply with helpful and concise answers which include driving action justification."""


class Chat:
    """ Helper class to manage a conversation with the OpenAI API over multiple turns. """

    def __init__(self,
                 system_prompt: str = None,
                 auto_log: bool = True):
        """ Initialize the Chat class.

        Args:
            system_prompt: The system prompt to use for the chat.
                Uses car explainer system prompt by default.
            auto_log: Whether to automatically log the chat history ([True]/False).
        """
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self._client = openai.OpenAI(api_key=openai.api_key)

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        self._system = system_prompt
        if auto_log:
            logger.warning("[SYSTEM] %s", system_prompt)
        self._auto_log = auto_log

        self._user_history = []
        self._response_history = []

    def prompt(self,
               prompt: str,
               use_history: bool = True,
               **gpt_kwargs):
        """ Send a prompt to the OpenAI API and return the response.

        Args:
            prompt: The message to send to the API.
            use_history: Whether to use the chat history in the prompt.

        Keyword Args:
            gpt_kwargs: Additional arguments to pass to the OpenAI API.
        """
        # Setup message history if needed
        messages = [
            {"role": "system", "content": self._system},
        ] + self.history if use_history else []

        # Create user message
        user_message = {"role": "user", "content": prompt}
        messages.append(user_message)
        self._user_history.append(user_message)

        if self._auto_log:
            logger.warning("\n[USER] %s", prompt)

        # Send message to API and store response if not empty
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            **gpt_kwargs
        )

        self.response_history.append(response)

        if self._auto_log:
            logger.warning("\n[ASSISTANT] %s", response.choices[0].message.content)

        return response.choices[0].message.content

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
    def client(self) -> openai.Client:
        """ Return the OpenAI client. """
        return self._client

    @property
    def system_prompt(self) -> str:
        """ Return the system prompt. """
        return self._system

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
