# Notification API Delivery Semantics
## Delivery
The API accepts requests and publishes delivery jobs asynchronously.
## Idempotency
Clients should provide a request_id to avoid duplicate delivery jobs.
## Channels
Email, SMS, and push are supported.
## Status
The synchronous response confirms acceptance, not final delivery. Use the status endpoint or webhook for completion.
