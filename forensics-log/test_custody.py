import sqlite3
from custody_log import init_db, log_event, verify_chain

init_db()
log_event("EV-001", "FILE_RECEIVED", "abc123", "Uploaded by officer")
log_event("EV-001", "AI_ANALYSIS_RUN", "abc123", "Risk score 0.91")
log_event("EV-001", "CERTIFICATE_GENERATED", "abc123", "Draft created")

print("Before tampering:", verify_chain())      # expect (True, None)

conn = sqlite3.connect("custody.db")
conn.execute("UPDATE custody_log SET details='Risk score 0.05' WHERE id=2")
conn.commit()
conn.close()

print("After tampering:", verify_chain())       # expect (False, 2)