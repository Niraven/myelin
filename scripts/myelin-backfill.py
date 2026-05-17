#!/usr/bin/env python3
"""Myelin Backfill — populates Myelin episodes from Hermes session history.

Reads all past sessions from the Hermes state DB, extracts tool calls and
their results, and batch-observes them into Myelin via the ObservationQueue.

Usage:
    python scripts/myelin-backfill.py [--limit 1000] [--dry-run]

Run once after fresh Myelin install. Takes ~60s for 500 episodes.
"""

import argparse
import json
import os
import sqlite3
import sys
import time

# Add parent so we can import myelin
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def get_hermes_sessions(db_path: str, limit: int = 0) -> list[dict]:
    """Read sessions and messages from Hermes state DB."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get recent sessions
    query = "SELECT id, started_at FROM sessions ORDER BY started_at DESC"
    if limit:
        query += f" LIMIT {limit}"
    sessions = [dict(r) for r in conn.execute(query).fetchall()]

    result = []
    for session in sessions:
        messages = conn.execute(
            "SELECT role, content, tool_calls, tool_name, timestamp "
            "FROM messages WHERE session_id = ? ORDER BY timestamp",
            (session["id"],),
        ).fetchall()
        result.append(
            {
                "session_id": session["id"],
                "timestamp": session["started_at"],
                "messages": [dict(m) for m in messages],
            }
        )

    conn.close()
    return result


def extract_tool_episodes(session: dict) -> list[dict]:
    """Extract tool call + result pairs as episodes."""
    episodes = []
    messages = session["messages"]

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant" or not msg["tool_calls"]:
            continue

        try:
            tool_calls = json.loads(msg["tool_calls"])
        except (json.JSONDecodeError, TypeError):
            continue

        for tc in tool_calls:
            func = tc.get("function", {})
            tool_name = func.get("name", "unknown")
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}

            # Find matching tool response
            result_content = ""
            for j in range(i + 1, min(i + 5, len(messages))):
                if (
                    messages[j]["role"] == "tool"
                    and messages[j].get("tool_call_id") == tc.get("id")
                ):
                    result_content = messages[j]["content"] or ""
                    break

            # Build episode
            content = _build_episode_content(tool_name, args, result_content)
            if content:
                episodes.append(
                    {
                        "action": tool_name,
                        "action_type": "tool_call",
                        "content_text": content,
                        "session_id": session["session_id"],
                        "domain": _infer_domain(tool_name),
                        "success": _is_success(result_content),
                    }
                )

    return episodes


def _build_episode_content(tool_name: str, args: dict, result: str) -> str:
    """Build a meaningful content string from tool call + result."""
    # Summarize the tool call
    if tool_name == "terminal":
        cmd = args.get("command", "")
        return f"Ran terminal command: {cmd[:200]}"
    elif tool_name == "web_search":
        query = args.get("query", "")
        return f"Searched web for: {query}"
    elif tool_name == "web_extract":
        urls = args.get("urls", [])
        return f"Extracted content from: {', '.join(urls[:3])}"
    elif tool_name == "read_file":
        path = args.get("path", "")
        return f"Read file: {path}"
    elif tool_name == "write_file":
        path = args.get("path", "")
        return f"Wrote file: {path}"
    elif tool_name == "patch":
        path = args.get("path", "")
        return f"Patched file: {path}"
    elif tool_name == "search_files":
        pattern = args.get("pattern", "")
        target = args.get("target", "content")
        return f"Searched {target} for: {pattern}"
    elif tool_name == "delegate_task":
        goal = args.get("goal", "")[:200]
        return f"Delegated task: {goal}"
    elif tool_name == "delegate_with_model":
        goal = args.get("goal", "")[:200]
        model = args.get("model", "")
        return f"Delegated to {model}: {goal}"
    elif tool_name == "memory":
        action = args.get("action", "")
        target = args.get("target", "")
        return f"Memory {action} ({target})"
    elif tool_name == "skill_manage":
        action = args.get("action", "")
        name = args.get("name", "")
        return f"Skill {action}: {name}"
    elif tool_name == "cronjob":
        action = args.get("action", "")
        return f"Cron job {action}"
    elif tool_name.startswith("mcp_myelin"):
        return ""  # Skip myelin tool calls (circular)
    elif tool_name.startswith("mcp_remio"):
        return ""  # Skip remio tool calls
    else:
        # Generic fallback
        arg_summary = ", ".join(f"{k}={str(v)[:50]}" for k, v in list(args.items())[:3])
        return f"Called {tool_name}({arg_summary})"


def _infer_domain(tool_name: str) -> str:
    """Infer domain from tool name."""
    domain_map = {
        "terminal": "devops",
        "web_search": "research",
        "web_extract": "research",
        "read_file": "development",
        "write_file": "development",
        "patch": "development",
        "search_files": "development",
        "delegate_task": "orchestration",
        "delegate_with_model": "orchestration",
        "memory": "memory",
        "skill_manage": "skills",
        "cronjob": "automation",
        "browser_navigate": "web",
        "browser_click": "web",
        "send_message": "messaging",
    }
    return domain_map.get(tool_name, "general")


def _is_success(result: str) -> bool:
    """Check if a tool result looks successful."""
    if not result:
        return True
    result_lower = result.lower()
    return "error" not in result_lower[:100] or "traceback" not in result_lower[:200]


def main():
    parser = argparse.ArgumentParser(description="Backfill Myelin from Hermes sessions")
    parser.add_argument("--limit", type=int, default=50, help="Max sessions to process")
    parser.add_argument("--dry-run", action="store_true", help="Print episodes without observing")
    parser.add_argument(
        "--db", default=os.path.expanduser("~/.hermes/state.db"), help="Hermes state DB path"
    )
    parser.add_argument(
        "--myelin-db",
        default=os.path.expanduser("~/.hermes/data/myelin-hermes.db"),
        help="Myelin DB path",
    )
    args = parser.parse_args()

    print(f"Reading sessions from {args.db}...")
    sessions = get_hermes_sessions(args.db, limit=args.limit)
    print(f"Found {len(sessions)} sessions")

    all_episodes = []
    for session in sessions:
        episodes = extract_tool_episodes(session)
        all_episodes.extend(episodes)

    print(f"Extracted {len(all_episodes)} tool episodes")

    if args.dry_run:
        print("\n--- DRY RUN (first 10 episodes) ---")
        for ep in all_episodes[:10]:
            print(f"  [{ep['action']}] {ep['content_text'][:80]}")
        print(f"\nTotal: {len(all_episodes)} episodes (not observed)")
        return

    # Batch observe into Myelin
    from myelin.core.database import Database
    from myelin.ingest import Observation, ObservationQueue

    db = Database(args.myelin_db)
    queue = ObservationQueue(db, flush_interval_s=2.0, batch_size=100)

    print(f"Observing {len(all_episodes)} episodes into Myelin...")
    start = time.time()

    for ep in all_episodes:
        obs = Observation.from_observe_call(
            agent_id="hermes",
            agent_profile="hermes",
            action=ep["action"],
            action_type=ep["action_type"],
            content_text=ep["content_text"],
            session_id=ep["session_id"],
            domain=ep["domain"],
            success=ep["success"],
        )
        try:
            queue.enqueue(obs)
        except Exception:
            pass

    flushed = queue.flush_all()
    elapsed = time.time() - start

    print(f"Done! Flushed {flushed} episodes in {elapsed:.1f}s")
    print(f"Rate: {flushed / elapsed:.0f} episodes/sec")

    # Run sleep cycle to build graph
    print("\nRunning sleep cycle to build entity graph...")
    import asyncio
    from myelin.cognitive.sleep import SleepCycle

    sleep = SleepCycle(db)
    sleep_result = asyncio.run(sleep.run())
    print(f"  Relationships created: {sleep_result.get('relationships_created', 0)}")
    print(f"  Temporal states updated: {sleep_result.get('temporal_states_updated', 0)}")
    print(f"  Entities processed: {sleep_result.get('nrem', {}).get('entities_processed', 0)}")

    # Run promoter
    print("\nRunning promoter to discover procedures...")
    from myelin.cognitive.promoter import Promoter
    from myelin.memory.episodic import EpisodicMemory
    from myelin.memory.procedural import ProceduralMemory

    episodic = EpisodicMemory(db)
    procedural = ProceduralMemory(db)
    promoter = Promoter(db, episodic, procedural)
    promo_result = asyncio.run(promoter.run())
    print(f"  Procedures promoted: {promo_result.get('created', 0)}")

    # Final stats
    stats = db.fetchone("SELECT COUNT(*) as cnt FROM episodes")
    entities = db.fetchone("SELECT COUNT(*) as cnt FROM entities")
    rels = db.fetchone("SELECT COUNT(*) as cnt FROM relationships")
    procs = db.fetchone("SELECT COUNT(*) as cnt FROM procedures WHERE status='active'")

    print(f"\n--- Myelin Stats ---")
    print(f"  Episodes: {stats['cnt']}")
    print(f"  Entities: {entities['cnt']}")
    print(f"  Relationships: {rels['cnt']}")
    print(f"  Active procedures: {procs['cnt']}")
    print(f"\nMyelin is ready! Next session will use learned procedures.")


if __name__ == "__main__":
    main()
