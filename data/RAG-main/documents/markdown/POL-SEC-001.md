# Engineering Secrets Management Policy
## Approved Storage
Secrets must be stored in the organization's secret manager. Environment variables may be used at runtime, but secrets must not be committed to source control.
## Logging
API keys, access tokens, passwords, private keys, and database credentials must never appear in logs.
## Rotation
Production credentials must be rotated immediately after suspected exposure and according to the service-specific schedule.
