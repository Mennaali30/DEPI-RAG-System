# Blue-Green Deployment Guide
## Model
Two production environments exist: blue and green. One serves traffic while the other receives the candidate release.
## Process
Deploy the candidate to the inactive environment, run health checks and smoke tests, then switch traffic.
## Rollback
Rollback is a traffic switch to the previously active environment, provided database compatibility is maintained.
