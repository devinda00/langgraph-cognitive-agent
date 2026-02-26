# main.py
"""
Cognitive Agent — threaded architecture.

• Terminal 1  (this one):   User CLI  — type messages, see agent responses.
• Terminal 2  (run `tail -f agent.log`):   Live agent logs.

The agent graph runs on a background daemon thread.
The user CLI runs on the main thread.
They communicate through a thread-safe message bus.
"""
import asyncio
import threading
import sys

from dotenv import load_dotenv
load_dotenv()

from agent.logging_config import setup_logging, LOG_FILE, get_logger
from agent.message_bus import bus, MessageType, Message
from agent.graphs.mind import create_agent_graph
from agent.permanent_knowledge import PERMANENT_KNOWLEDGE

log = get_logger("main")


# ─────────────────────────────── Agent Thread ────────────────────────────────

def agent_thread_entry():
    """Entry point for the background agent thread. Runs the async event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_agent_loop())
    except KeyboardInterrupt:
        log.info("Agent loop interrupted.")
    except Exception as exc:
        log.exception("Agent loop crashed: %s", exc)
        bus.send_agent_response(f"[ERROR] Agent crashed: {exc}", sender="System")
    finally:
        loop.close()
        log.info("Agent thread exiting.")


async def _run_agent_loop():
    """The actual agent graph execution loop (runs inside the agent thread)."""
    app = create_agent_graph()
    PERMANENT_KNOWLEDGE.add_memory("Clawed is a helpful AI assistant created by Devinda.")

    log.info("Agent graph compiled and ready.")
    bus.send_agent_response("Agent is ready. Type your message below.", sender="System")

    initial_state = {
        "messages": [],
        "temp_knowledge": {},
        "permanent_knowledge": PERMANENT_KNOWLEDGE,
        "has_new_input": False,
        "generated_questions": [],
        "brain_thought": None,
        "mind_action": None,
    }

    config = {"recursion_limit": 1000}

    try:
        async for step in app.astream(initial_state, config):
            if bus.is_shutdown:
                break
            step_name = list(step.keys())[0]
            log.info("Executing step: %s", step_name)
    except KeyboardInterrupt:
        log.info("Agent received shutdown signal.")


# ───────────────────────────────── User CLI ──────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RESET = "\033[0m"

PROMPT = f"{CYAN}{BOLD}You ❯ {RESET}"


def response_listener():
    """
    Background thread that drains agent responses and prints them.
    No ANSI cursor manipulation — just plain prints on new lines.
    Status messages are NOT shown here (only in agent.log).
    """
    while not bus.is_shutdown:
        msg = bus.get_agent_response(timeout=0.3)
        if msg is None:
            continue
        if msg.type == MessageType.AGENT_RESPONSE:
            # Print on a new line. This may briefly interrupt the user's
            # typing visually, but input() on the main thread still
            # captures their text correctly.
            print(f"\n{GREEN}{BOLD}[{msg.sender}]{RESET} {msg.content}", flush=True)
        # STATUS, ERROR, etc. are silently consumed (visible in agent.log)


def cli_loop():
    """
    Main-thread CLI.  Uses input() for reliable readline support.
    A background listener thread prints agent responses as they arrive.
    The listener does NO cursor manipulation or prompt reprinting, so
    there is no conflict with input().
    """
    # ── Wait for the agent to signal readiness ──
    while True:
        msg = bus.get_agent_response(timeout=0.5)
        if msg is not None and msg.type == MessageType.AGENT_RESPONSE:
            print(f"{GREEN}{BOLD}[{msg.sender}]{RESET} {msg.content}")
            break

    print(f"{DIM}(Agent logs → run  tail -f {LOG_FILE}  in another terminal){RESET}")
    print(f"{DIM}The agent runs autonomously. Type a message any time, or 'exit' to stop.{RESET}\n")

    # Start the response listener on a daemon thread
    listener = threading.Thread(target=response_listener, daemon=True, name="cli-listener")
    listener.start()

    try:
        while not bus.is_shutdown:
            try:
                user_input = input(PROMPT)
            except EOFError:
                break

            text = user_input.strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit"):
                break

            bus.send_user_message(text)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n{DIM}Shutting down…{RESET}")
        bus.request_shutdown()


# ────────────────────────────────── Main ─────────────────────────────────────

def main():
    setup_logging()
    log.info("Starting cognitive agent (threaded mode)")

    # Launch the agent on a daemon thread so it dies with the process
    agent = threading.Thread(target=agent_thread_entry, daemon=True, name="agent")
    agent.start()

    # Run the CLI on the main thread (handles Ctrl-C naturally)
    cli_loop()

    # Give the agent thread a moment to clean up
    agent.join(timeout=3)
    print("Goodbye.")


if __name__ == "__main__":
    main()

