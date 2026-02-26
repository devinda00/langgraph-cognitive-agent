# agent/message_bus.py
"""
Thread-safe message bus for communication between the user CLI and the agent.

The CLI thread puts user messages into `user_input_queue`.
The agent thread reads from `user_input_queue` and puts responses into `agent_response_queue`.
The CLI thread reads from `agent_response_queue` to display responses.
"""
import queue
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MessageType(Enum):
    USER_MESSAGE = "user_message"
    AGENT_RESPONSE = "agent_response"
    AGENT_STATUS = "agent_status"      # e.g. "thinking...", "searching..."
    SHUTDOWN = "shutdown"
    ERROR = "error"


@dataclass
class Message:
    type: MessageType
    content: str
    sender: str = ""


class MessageBus:
    """Thread-safe message bus connecting the CLI and agent threads."""

    def __init__(self):
        self.user_input_queue: queue.Queue[Message] = queue.Queue()
        self.agent_response_queue: queue.Queue[Message] = queue.Queue()
        self._shutdown_event = threading.Event()

    # --- CLI side ---

    def send_user_message(self, text: str):
        """Called by the CLI thread to send a message to the agent."""
        self.user_input_queue.put(Message(MessageType.USER_MESSAGE, text, sender="User"))

    def get_agent_response(self, timeout: float = 0.5) -> Optional[Message]:
        """Called by the CLI thread to receive a response from the agent. Non-blocking."""
        try:
            return self.agent_response_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # --- Agent side ---

    def wait_for_user_input(self) -> Message:
        """
        Called by the agent thread. Blocks until the user sends a message
        or a shutdown is requested.
        """
        while not self._shutdown_event.is_set():
            try:
                msg = self.user_input_queue.get(timeout=0.5)
                return msg
            except queue.Empty:
                continue
        return Message(MessageType.SHUTDOWN, "")

    def try_get_user_input(self, timeout: float = 0.1) -> Optional[Message]:
        """
        Non-blocking check for user input.
        Returns the message if one is available, otherwise None.
        """
        try:
            return self.user_input_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def send_agent_response(self, text: str, sender: str = "Mind"):
        """Called by the agent thread to send a response to the CLI."""
        self.agent_response_queue.put(
            Message(MessageType.AGENT_RESPONSE, text, sender=sender)
        )

    def send_agent_status(self, text: str):
        """Called by the agent thread to send a status update to the CLI."""
        self.agent_response_queue.put(
            Message(MessageType.AGENT_STATUS, text)
        )

    # --- Lifecycle ---

    def request_shutdown(self):
        """Signal both threads to shut down gracefully."""
        self._shutdown_event.set()
        # Push a sentinel so the agent isn't stuck waiting
        self.user_input_queue.put(Message(MessageType.SHUTDOWN, ""))

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown_event.is_set()


# Global singleton — imported by both the CLI and agent modules
bus = MessageBus()
