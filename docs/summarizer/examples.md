# Held-out examples: base vs fine-tuned

### Example 1
**Log:** [CRIT] node-08: volume /var/log at 97%; writes from auth-gateway are failing in connectDb
**Reference:** Writes from auth-gateway on node-08 are failing because volume /var/log is 97% full — connectDb hit 'no space left on device'.
**Base t5-small:** connectDb is failing to connectDb.
**Fine-tuned:** volume /var/log on node-08 is 97% higher than auth-gateway's connectDb.

### Example 2
**Log:** [CRIT] payment-worker: 401 on /api/v1/orders — expired credentials (157m past validity)
**Reference:** An expired token (157 minutes past validity) caused payment-worker to reject /api/v1/orders with 401.
**Base t5-small:** expired credentials (157m past validity) expired credentials.
**Fine-tuned:** Payment-worker failed to process /api/v1/orders with 401 (exit 157 minutes past validity).

### Example 3
**Log:** FATAL service=media-encoder fn=acquireLock missing_env=MAX_WORKERS exit=1
**Reference:** A missing MAX_WORKERS value stopped media-encoder from booting (acquireLock exited 1).
**Base t5-small:** fn=acquireLock missing_env=MAX_WORKERS exit=1.
**Fine-tuned:** Media-encoder failed to start because the MAX_WORKERS database address was missing; the acquireLock call failed with exit code 1.

### Example 4
**Log:** Access denied for role app_writer when email-dispatcher executed openChannel against table invoices. Error 42501.
**Reference:** Error 42501 in openChannel: the app_writer role used by email-dispatcher has no access to invoices.
**Base t5-small:** access denied for role app_writer when email-dispatcher executed openChannel against table invoices. error 42501.
**Fine-tuned:** email-dispatcher failed to access table invoices because role app_writer was denied access to table invoices; error 42501 in openChannel.

### Example 5
**Log:** [CRIT] thumbnail-worker on node-05: lookup of notify-hub returned SERVFAIL
**Reference:** thumbnail-worker on node-05 cannot resolve its upstream notify-hub — DNS returned SERVFAIL.
**Base t5-small:** notify-hub returned SERVFAIL. a thumbnail-worker on node-05 was found on node-05.
**Fine-tuned:** thumbnail-worker on node-05 was missing because notify-hub had no access to SERVFAIL.
