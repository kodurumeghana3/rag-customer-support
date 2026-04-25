import sqlite3
import uuid
from datetime import datetime
from typing import Optional

class HITLManager:
    def __init__(self, db_path="./hitl_tickets.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id  TEXT PRIMARY KEY,
                query      TEXT,
                context    TEXT,
                intent     TEXT,
                confidence REAL,
                status     TEXT DEFAULT 'pending',
                created_at TEXT,
                response   TEXT,
                resolved_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def create_ticket(self, query, context="", intent="unknown", confidence=0.0):
        ticket_id = "T-" + str(uuid.uuid4())[:8].upper()
        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO tickets
               (ticket_id, query, context, intent, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ticket_id, query, context, intent, confidence, now)
        )
        conn.commit()
        conn.close()
        print(f"[HITL] Ticket created: {ticket_id}")
        return ticket_id

    def get_ticket(self, ticket_id):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def resolve_ticket(self, ticket_id, human_response):
        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(
            """UPDATE tickets SET status='resolved',
               response=?, resolved_at=? WHERE ticket_id=?""",
            (human_response, now, ticket_id)
        )
        conn.commit()
        conn.close()
        return cur.rowcount > 0

    def list_tickets(self, status=None):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if status:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE status=? ORDER BY created_at DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tickets ORDER BY created_at DESC"
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]