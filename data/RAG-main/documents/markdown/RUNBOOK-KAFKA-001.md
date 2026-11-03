# Kafka Consumer Lag Runbook
## Symptoms
Consumer lag increases when consumers process records slower than producers publish them.
## First Checks
Check consumer group lag, partition assignment, consumer errors, and downstream dependency latency.
## Common Causes
Slow database operations, insufficient consumers, partition skew, and downstream throttling.
## Mitigation
Scale consumers only within the partition limit. If the downstream dependency is the bottleneck, scaling consumers may worsen the problem.
