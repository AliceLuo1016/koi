# Log Monitor Skill

Monitors application logs for errors and anomalies. Creates alerts for issues found.

## Workflow

1. Generate or read logs from `mock/logs/latest.log`
2. Analyze for ERROR and WARN patterns
3. Group errors by service and type
4. Create alerts for critical issues (>3 errors from same service, or any critical error)
5. Skip if no significant issues found

## Usage

When asked to monitor logs or when run as a cron task:

```
1. Read mock/logs/latest.log
2. Count errors by service
3. If any service has 3+ errors → create_alert with severity "high"
4. If any critical keywords (out of memory, disk critical, circuit breaker) → create_alert with severity "critical"  
5. Summarize findings
```
