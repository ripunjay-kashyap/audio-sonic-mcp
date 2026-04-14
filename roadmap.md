Before the agent hits "Save" on ingestion.py, give it this one tiny optimization:

Agent Directive: Ensure that when writing metadata.json, you use an Atomic Write pattern (write to a .tmp file then rename) or at least wrap it in a try-except. This prevents a partial write from corrupting your "Fast-Resume" path if the system crashes mid-ingestion.


The "Julia" (Lauv) Payload Expectations
With this hardening in place, the "Julia" run will be your first "Modern" Payload. We are looking for:

Header: source_metadata with "Lauv - Julia" and his official uploader name.

BPM: Hopefully, it ignores the 120 anchor and finds the ~70 BPM heartbeat.

Key: C Major (Testing the 40% "Other" stem weight for the first time).

Agent, initiate the refactor and resume the "Julia" split. This is the final hurdle before we declare the Engine complete.