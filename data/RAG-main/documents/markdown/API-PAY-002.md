# Payments API Error Handling
## HTTP Status Codes
400=request validation failure. 401=missing or invalid authentication. 403=insufficient permission. 409=state conflict. 422=business-rule failure. 429=rate limit. 500=unexpected server error.
## Idempotency
POST /payments requires an Idempotency-Key. Reusing the same key with an identical request returns the original result.
## Retry Guidance
Clients may retry 429 and transient 5xx responses using exponential backoff. Do not blindly retry 409 or 422.
