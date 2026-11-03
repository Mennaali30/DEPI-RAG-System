# Troubleshooting — HTTP 429 Responses
## Meaning
HTTP 429 indicates that a client exceeded an applicable rate limit.
## Diagnosis
Inspect Retry-After, gateway metrics, client request rate, and whether multiple workers share the same client identity.
## Correct Response
Back off according to Retry-After when provided, otherwise use exponential backoff.
## Do Not
Do not increase concurrency to compensate for 429 responses.
