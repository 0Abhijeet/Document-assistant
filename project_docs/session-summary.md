# AWS migration of Document Assistant

Full record of what was actually done in this session, in order.

## Decision made
Chose to migrate Postgres from Neon to **RDS** (rather than keep Neon + only use EC2), on the basis that RDS is explicitly named in target JDs and the VPC/security-group work involved is itself the transferable skill, not just a keyword match.

## Phase 0 — AWS account setup
- Created IAM user `abhijeeth-admin` with `AdministratorAccess`, console access enabled (had to be fixed after initially creating without console access)
- Confirmed no access keys created/left on the account
- Enabled MFA on root
- Created a $5 cost budget with an alert threshold at 75%
- Confirmed account is on AWS's newer credit-based free tier ($100 Free Tier credit + $20 Budgets credit, $120 total, not the older "750 hrs/month" model)

## Phase 1 — EC2 launch
- Instance: `document-assistant-prod`, **t2.micro** (free-tier eligible), Ubuntu Server 22.04 LTS
- Fixed a wrong-AMI selection (had picked a SQL-Server-bundled Ubuntu image by mistake) by clearing a stuck search filter
- Key pair created as `.ppk` by mistake; abandoned the local-SSH-client path entirely and used **EC2 Instance Connect** (browser-based) instead
- Security group `launch-wizard-3`: removed a stray MSSQL rule, opened HTTP/HTTPS/port 8000, SSH temporarily opened to `0.0.0.0/0` to unblock Instance Connect (**still open — deliberately deferred**, see Open items)
- Set up a 2GB swap file (`/swapfile`) since t2.micro's 1GB RAM is tight for the fastembed/ONNX embedding model

## Phase 2 — Docker deployment
- Installed Docker via the official Docker apt repo (Ubuntu's default repo doesn't carry `docker-compose-plugin`)
- Cloned the repo, found the existing `docker-compose.yml` also defines a local `db` (Postgres) service used for CI testing — not meant for this deployment
- Ran only the `app` service against Neon initially, using `.env` + `docker compose up -d --no-deps app`
- Verified end-to-end against Neon via browser at `http://<ec2-ip>:8000/docs`

## Phase 3 — RDS + migration
- Created `document-assistant-db`: PostgreSQL 16.x, **db.t3.micro**, 20GB storage, self-managed credentials, Single-AZ, no public access, new security group `document-assistant-rds-sg`
- Avoided an accidental Aurora Serverless "express create" path and an oversized default instance class (`db.m7g.large`) along the way
- Enabled `pgvector` extension via psql
- Migrated data from Neon: `pg_dump` → `pg_restore --no-owner --no-privileges` — confirmed 2 documents / 14 chunks landed correctly
- Repointed the app's `DATABASE_URL` to RDS, restarted the container, verified end-to-end via browser again

## Phase 5 — Persistence
- Verified via a real instance reboot that the app container comes back on its own (`restart: unless-stopped`), no manual intervention needed

## Networking hardening
- Allocated and associated an **Elastic IP** (`15.135.171.245`) so the demo URL survives stop/start cycles
- Diagnosed and fixed RDS connectivity: the RDS security group's inbound rule was initially IP-based (and briefly widened to `0.0.0.0/0` as a bad shortcut, then corrected) — fixed by referencing the EC2 security group directly (SG-to-SG), the correct pattern for private service-to-service access

## Scope decision
Explicitly decided **against** Phase 6 (Nginx/HTTPS) — reasoned that it demonstrates general webserver skills, not AWS-specific ones, and doesn't address the stated goal (closing the AWS resume/JD gap). Render remains the actual long-term demo; this AWS deployment is a resume/interview artifact, not a permanent service.

## Documentation produced
- `aws-migration-plan.md` — full phase-by-phase plan with a running debug log (6 real issues logged with symptom/cause/fix/lesson)
- `README-aws-section.md` — architecture summary, skills demonstrated, and issue log, written for direct inclusion in the project's GitHub README
- Architecture diagram (VPC / EC2 / Docker / RDS / security groups) rendered in-chat

## Open items (not yet done, by your choice)
1. **SSH security group rule still `0.0.0.0/0`** — deliberately deferred to end of project; close by scoping back to "My IP" and re-verifying EC2 Instance Connect still works
2. **Eventual teardown** — release the Elastic IP and stop/delete EC2 + RDS once documentation is captured, so nothing runs forgotten against the $120 credit balance
3. **Resume line drafted but not yet confirmed placed**: *"Deployed RAG application to AWS EC2 with RDS PostgreSQL (pgvector); configured VPC security groups, IAM, and Docker with persistent restart policies."*
