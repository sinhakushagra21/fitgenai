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
import json
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage

from agent import create_graph
from agent.persistence import get_context_state, get_latest_context_state_by_email, init_db


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
    init_db()
    graph = create_graph()

    # ── Conversation state ───────────────────────────────────────
    user_email = os.getenv("FITGEN_USER_EMAIL", "").strip()
    context_id = os.getenv("FITGEN_CONTEXT_ID", str(uuid.uuid4()))
    restored = get_context_state(context_id) or get_latest_context_state_by_email(user_email) or {}
    context_id = restored.get("context_id", context_id)
    state = {
        "messages": [],
        "user_profile": restored.get("user_profile", {}),
        "user_email": user_email or restored.get("user_email", ""),
        "context_id": context_id,
        "state_id": context_id,
        "workflow": restored.get("workflow", {}),
        "calendar_sync_requested": restored.get("calendar_sync_requested", False),
    }

    print(f"🔖  Context ID: {context_id}")

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
        tool_direct_reply = False
        for event in graph.stream(state, stream_mode="values"):
            if event.get("messages"):
                last_msg = event["messages"][-1]

                if isinstance(last_msg, ToolMessage):
                    if last_msg.content:
                        try:
                            payload = json.loads(last_msg.content)
                            assistant_message = payload.get("assistant_message")
                            if assistant_message and assistant_message != response_content:
                                print(assistant_message, end="", flush=True)
                                response_content = assistant_message
                                tool_direct_reply = True
                        except json.JSONDecodeError:
                            pass
                    continue

                # Skip AIMessages that contain tool_calls (routing decisions)
                if getattr(last_msg, "tool_calls", None):
                    continue

                # If tool already provided the user-facing content, ignore relay ack.
                if tool_direct_reply:
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
        if "user_profile" in event:
            state["user_profile"] = event["user_profile"]
        if "user_email" in event:
            state["user_email"] = event["user_email"]
        if "workflow" in event:
            state["workflow"] = event["workflow"]
        if "context_id" in event:
            state["context_id"] = event["context_id"]
        if "state_id" in event:
            state["state_id"] = event["state_id"]
        if "calendar_sync_requested" in event:
            state["calendar_sync_requested"] = event["calendar_sync_requested"]


if __name__ == "__main__":
    main()
