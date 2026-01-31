# Heartbeat message for safety watchdog
# Sent by client at 50Hz to maintain connection

# Timestamp in nanoseconds (system time)
uint64 timestamp_ns

# Sequence number for detecting dropped messages
uint32 sequence_number

# Unique identifier for the client
string client_id

# Optional: Client-reported latency for diagnostics (ms)
float32 reported_latency_ms
