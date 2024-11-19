""" Helper classes and functions for interacting with the OpenAI API over multiple turns. """

import logging
import json
from itertools import chain
from typing import List, Dict
import openai


logger = logging.getLogger(__name__)


class Chat:
    """ Helper class to manage a conversation with the OpenAI API over multiple turns. """

    def __init__(self, system_prompt: str = None):
        """ Initialize the Chat class."""
        self._client = openai.OpenAI(api_key=openai.api_key)

        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        self._system = system_prompt
        self._user_history = []
        self._assistant_history = []

    def send(self, prompt: str, use_history: bool = True, **gpt_kwargs):
        """ Send a message to the OpenAI API and return the response. 
        
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

        # Send message to API and store response if not empty
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            **gpt_kwargs
        )
        response = response.choices[0].message.content
        if response:
            self.assistant_history.append({"role": "assistant", "content": response})
        else:
            logger.warning("Empty response from OpenAI API.")

        return response

    def reset_history(self):
        """ Reset all of the chat history. """
        self._user_history = []
        self._assistant_history = []

    def save_chat_history(self, path: str):
        """ Save the chat history to a file. """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f)

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
        return list(chain.from_iterable(zip(self._user_history, self._assistant_history)))

    @property
    def user_history(self) -> List[Dict[str, str]]:
        """ Return the message history of the user. """
        return self._user_history

    @property
    def assistant_history(self) -> List[Dict[str, str]]:
        """ Return the message history of the assistant. """
        return self._assistant_history
