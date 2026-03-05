"""
Daily goals tracker — SQLite-backed storage for 3-goal daily habit loop.

Tables:
  daily_goals   — one row per goal per day per user
  user_chat_ids — maps Telegram user_id → chat_id for proactive messaging
"""

import os
import sqlite3
import threading
import time
from datetime import date

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_PROJECT_DIR, "data", "goals.db")
_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(_DB_PATH)


def init_db() -> None:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    with _lock:
        conn = _conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS daily_goals (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT    NOT NULL,
                date         TEXT    NOT NULL,
                goal_number  INTEGER NOT NULL,
                goal_text    TEXT    NOT NULL,
                completed    INTEGER NOT NULL DEFAULT 0,
                created_at   REAL    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_chat_ids (
                user_id   TEXT    PRIMARY KEY,
                chat_id   INTEGER NOT NULL,
                username  TEXT
            );
        """)
        conn.commit()
        conn.close()


init_db()

# ---------------------------------------------------------------------------
# Chat ID registry — needed for proactive (scheduler-initiated) messages
# ---------------------------------------------------------------------------

def register_user(user_id: int, chat_id: int, username: str = None) -> None:
    """Store user's chat_id so the scheduler can send proactive messages."""
    with _lock:
        conn = _conn()
        conn.execute(
            "INSERT OR REPLACE INTO user_chat_ids (user_id, chat_id, username) VALUES (?, ?, ?)",
            (str(user_id), chat_id, username)
        )
        conn.commit()
        conn.close()


def get_all_chat_ids() -> list[dict]:
    """Return all registered users for broadcast scheduler jobs."""
    with _lock:
        conn = _conn()
        rows = conn.execute("SELECT user_id, chat_id, username FROM user_chat_ids").fetchall()
        conn.close()
    return [{"user_id": r[0], "chat_id": r[1], "username": r[2]} for r in rows]


# ---------------------------------------------------------------------------
# Goal CRUD
# ---------------------------------------------------------------------------

def _today() -> str:
    return date.today().isoformat()


def save_goals(user_id: int, goals: list[str]) -> dict:
    """
    Save today's goals for a user (replaces any existing goals for today).
    goals: list of up to 3 strings.
    """
    today = _today()
    with _lock:
        conn = _conn()
        # Clear existing goals for today
        conn.execute(
            "DELETE FROM daily_goals WHERE user_id = ? AND date = ?",
            (str(user_id), today)
        )
        for i, text in enumerate(goals[:3], start=1):
            conn.execute(
                "INSERT INTO daily_goals (user_id, date, goal_number, goal_text, completed, created_at) "
                "VALUES (?, ?, ?, ?, 0, ?)",
                (str(user_id), today, i, text.strip(), time.time())
            )
        conn.commit()
        conn.close()
    return {"saved": len(goals[:3]), "goals": goals[:3]}


def get_today_goals(user_id: int) -> list[dict]:
    """Return today's goals for a user as a list of dicts."""
    with _lock:
        conn = _conn()
        rows = conn.execute(
            "SELECT goal_number, goal_text, completed FROM daily_goals "
            "WHERE user_id = ? AND date = ? ORDER BY goal_number",
            (str(user_id), _today())
        ).fetchall()
        conn.close()
    return [{"number": r[0], "text": r[1], "completed": bool(r[2])} for r in rows]


def mark_goal_complete(user_id: int, goal_number: int, completed: bool = True) -> dict:
    """Mark a specific goal as complete (or incomplete)."""
    with _lock:
        conn = _conn()
        conn.execute(
            "UPDATE daily_goals SET completed = ? WHERE user_id = ? AND date = ? AND goal_number = ?",
            (1 if completed else 0, str(user_id), _today(), goal_number)
        )
        conn.commit()
        conn.close()
    return {"goal_number": goal_number, "completed": completed}


def format_goals_status(goals: list[dict]) -> str:
    """Return a readable string of today's goals with checkboxes."""
    if not goals:
        return "No goals set for today yet."
    lines = []
    for g in goals:
        icon = "✅" if g["completed"] else "⬜"
        lines.append(f"{icon} {g['number']}. {g['text']}")
    done = sum(1 for g in goals if g["completed"])
    lines.append(f"\n{done}/{len(goals)} complete")
    return "\n".join(lines)
