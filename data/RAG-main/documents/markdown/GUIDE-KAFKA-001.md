# Kafka Event Schema Guide
## Required Fields
Events must contain event_id, event_type, occurred_at, producer, and schema_version.
## Compatibility
Consumers must tolerate additive fields. Removing or changing the meaning of an existing field requires a new schema version.
## Idempotency
Consumers should use event_id to detect duplicate deliveries.
