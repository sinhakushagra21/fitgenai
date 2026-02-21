"""
app.py
──────
Entry-point for FITGEN.AI — runs a terminal-based conversational loop
that streams responses from the LangGraph agent.

Usage
-----
    $ cp .env.example .env   # fill in your API key
    $ pip install -r requirements.txt
    $ python app.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage

from agent import create_graph


def _print_banner() -> None:
    """Display the FITGEN.AI welcome banner."""
    banner = r"""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     ███████╗██╗████████╗ ██████╗ ███████╗███╗   ██╗          ║
║     ██╔════╝██║╚══██╔══╝██╔════╝ ██╔════╝████╗  ██║          ║
║     █████╗  ██║   ██║   ██║  ███╗█████╗  ██╔██╗ ██║          ║
║     ██╔══╝  ██║   ██║   ██║   ██║██╔══╝  ██║╚██╗██║          ║
║     ██║     ██║   ██║   ╚██████╔╝███████╗██║ ╚████║.AI       ║
║     ╚═╝     ╚═╝   ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═══╝         ║
║                                                               ║
║          Your AI-Powered Personal Fitness Coach               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print("  Type your fitness question or goal below.")
    print("  Type 'quit' or 'exit' to end the session.\n")


def main() -> None:
    """Run the FITGEN.AI conversational loop."""

    # ── Load environment variables ───────────────────────────────
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌  OPENAI_API_KEY not found. Copy .env.example → .env and add your key.")
        sys.exit(1)

    # ── Build graph ──────────────────────────────────────────────
    graph = create_graph()

    # ── Conversation state ───────────────────────────────────────
    state = {"messages": [], "user_profile": {}}

    _print_banner()

    # ── Main loop ────────────────────────────────────────────────
    while True:
        try:
            user_input = input("🧑  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋  Stay strong! See you next time.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("\n👋  Stay strong! See you next time.")
            break

        # Append user message to state
        state["messages"].append(HumanMessage(content=user_input))

        # Stream response
        print("\n🤖  FITGEN.AI: ", end="", flush=True)

        response_content = ""
        for event in graph.stream(state, stream_mode="values"):
            if event.get("messages"):
                last_msg = event["messages"][-1]

                # Skip ToolMessages (raw JSON from tools)
                if isinstance(last_msg, ToolMessage):
                    continue

                # Skip AIMessages that contain tool_calls (routing decisions)
                if getattr(last_msg, "tool_calls", None):
                    continue

                # Print only the final AI text response
                if (
                    hasattr(last_msg, "content")
                    and last_msg.content
                    and last_msg.content != response_content
                ):
                    new_text = last_msg.content[len(response_content):]
                    print(new_text, end="", flush=True)
                    response_content = last_msg.content

        print("\n")  # newline after streamed response

        # Update state with the full conversation (including AI reply)
        state["messages"] = event["messages"]


if __name__ == "__main__":
    main()
