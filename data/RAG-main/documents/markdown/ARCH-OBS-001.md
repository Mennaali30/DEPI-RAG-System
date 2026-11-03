# Observability Standards
## Required Signals
Production services must expose metrics, structured logs, and distributed traces.
## Correlation
Requests must propagate a correlation_id across service boundaries.
## Metrics
Record request count, error count, latency, and dependency failures.
## Logging
Logs should include timestamp, service, severity, and correlation_id. Sensitive customer data must not be logged.
