import hashlib
import sqlite3
from datetime import datetime, timezone

DB_PATH = "custody.db"
GENESIS_HASH = "0" * 64   # "previous hash" for the very first entry


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS custody_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            evidence_id TEXT NOT NULL,
            action      TEXT NOT NULL,
            file_hash   TEXT NOT NULL,
            details     TEXT NOT NULL,
            prev_hash   TEXT NOT NULL,
            entry_hash  TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def hash_file(path):
    """SHA-256 fingerprint of any file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_entry_hash(timestamp, evidence_id, action, file_hash, details, prev_hash):
    text = "|".join([timestamp, evidence_id, action, file_hash, details, prev_hash])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def log_event(evidence_id, action, file_hash, details=""):
    """(a) Add one event to the chain."""
    conn = get_connection()
    last = conn.execute(
        "SELECT entry_hash FROM custody_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    prev_hash = last[0] if last else GENESIS_HASH
    timestamp = datetime.now(timezone.utc).isoformat()
    entry_hash = compute_entry_hash(
        timestamp, evidence_id, action, file_hash, details, prev_hash
    )
    conn.execute(
        """INSERT INTO custody_log
           (timestamp, evidence_id, action, file_hash, details, prev_hash, entry_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (timestamp, evidence_id, action, file_hash, details, prev_hash, entry_hash),
    )
    conn.commit()
    conn.close()
    return entry_hash


def verify_chain():
    """(b) Check the whole chain. Returns (True, None) or (False, id_of_bad_row)."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT id, timestamp, evidence_id, action, file_hash, details,
                  prev_hash, entry_hash FROM custody_log ORDER BY id"""
    ).fetchall()
    conn.close()

    expected_prev = GENESIS_HASH
    for (row_id, ts, ev, act, fh, det, prev, eh) in rows:
        if prev != expected_prev:
            return False, row_id
        if compute_entry_hash(ts, ev, act, fh, det, prev) != eh:
            return False, row_id
        expected_prev = eh
    return True, None