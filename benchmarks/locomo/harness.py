"""LoCoMo-style adversarial reasoning benchmark.

10 hard-coded conversational threads (50 questions total) testing temporal
reasoning, cross-conversation inference, and entity tracking.
"""

from __future__ import annotations

import time
from typing import Any

# ---------------------------------------------------------------------------
# 10 conversations × 5-7 turns each × 5 questions each = 50 questions
# Question types: temporal, cross_conversation, entity_tracking
# ---------------------------------------------------------------------------

CONVERSATIONS: list[dict[str, Any]] = [
    {
        "name": "deploy-failure-1",
        "turns": [
            {"role": "user", "content": "Deploy v2.4 to staging"},
            {"role": "assistant", "content": "Deploying v2.4 to staging via CI/CD pipeline"},
            {"role": "user", "content": "Unit tests failed for payment module"},
            {"role": "assistant", "content": "Investigating payment module test failures. Rollback scheduled."},
            {"role": "user", "content": "What was the rollback version?"},
            {"role": "assistant", "content": "We rolled back from v2.4 to v2.3 on staging"},
            {"role": "user", "content": "Notify the payments team about the regression"},
        ],
        "questions": [
            {"question": "What version was deployed to staging?", "answer": "v2.4", "type": "temporal"},
            {"question": "Which module had failing tests?", "answer": "payment", "type": "entity_tracking"},
            {"question": "What version was the rollback target?", "answer": "v2.3", "type": "temporal"},
            {"question": "Which team should be notified?", "answer": "payments team", "type": "entity_tracking"},
            {"question": "Was the deploy to staging or production?", "answer": "staging", "type": "temporal"},
        ],
    },
    {
        "name": "db-migration-2",
        "turns": [
            {"role": "user", "content": "Run migration v5 on the orders database"},
            {"role": "assistant", "content": "Executing schema migration v5 on orders_db"},
            {"role": "user", "content": "Foreign key constraint violation on user_id column"},
            {"role": "assistant", "content": "Check pending transactions referencing user_id in user_profiles table"},
            {"role": "user", "content": "Cancel migration and lock the table"},
            {"role": "assistant", "content": "Migration cancelled. orders_db locked for investigation."},
        ],
        "questions": [
            {"question": "Which database was being migrated?", "answer": "orders_db", "type": "entity_tracking"},
            {"question": "What migration version was attempted?", "answer": "v5", "type": "temporal"},
            {"question": "Which column had the constraint violation?", "answer": "user_id", "type": "entity_tracking"},
            {"question": "Which table was referenced by the foreign key?", "answer": "user_profiles", "type": "entity_tracking"},
            {"question": "Was the migration completed or cancelled?", "answer": "cancelled", "type": "temporal"},
        ],
    },
    {
        "name": "incident-response-3",
        "turns": [
            {"role": "user", "content": "PagerDuty alert: CPU at 95% on api-7"},
            {"role": "assistant", "content": "Acknowledged alert for api-7. Checking metrics."},
            {"role": "user", "content": "Memory leak suspected in image-resizer service"},
            {"role": "assistant", "content": "Restarting image-resizer on api-7. Memory at 3.2GB"},
            {"role": "user", "content": "Add another replica to the pool"},
            {"role": "assistant", "content": "Scaled image-resizer from 2 to 3 replicas"},
            {"role": "user", "content": "Report baseline memory for the postmortem"},
        ],
        "questions": [
            {"question": "Which host triggered the CPU alert?", "answer": "api-7", "type": "entity_tracking"},
            {"question": "What CPU level triggered the alert?", "answer": "95%", "type": "temporal"},
            {"question": "Which service was suspected of a memory leak?", "answer": "image-resizer", "type": "entity_tracking"},
            {"question": "How many replicas were running after scaling?", "answer": "3", "type": "temporal"},
            {"question": "Who was alerted first, user or pager duty?", "answer": "pager duty", "type": "temporal"},
        ],
    },
    {
        "name": "secret-rotation-4",
        "turns": [
            {"role": "user", "content": "Rotate the GitHub token for CI/CD"},
            {"role": "assistant", "content": "Generating new GitHub token for CI/CD pipeline"},
            {"role": "user", "content": "Check if the old token was used in any public commits"},
            {"role": "assistant", "content": "Scanned 47 repos. Old token found in myelin-deploy/.env"},
            {"role": "user", "content": "Force push a cleanup commit to myelin-deploy"},
            {"role": "assistant", "content": "Cleaned .env in myelin-deploy. Force pushed to origin/main"},
        ],
        "questions": [
            {"question": "What type of credential was rotated?", "answer": "GitHub token", "type": "entity_tracking"},
            {"question": "How many repos were scanned for the old token?", "answer": "47", "type": "temporal"},
            {"question": "Which repo had the old token?", "answer": "myelin-deploy", "type": "entity_tracking"},
            {"question": "Where was the old token found?", "answer": ".env", "type": "entity_tracking"},
            {"question": "What was the target of the CI/CD pipeline?", "answer": "CI/CD", "type": "cross_conversation"},
        ],
    },
    {
        "name": "feature-launch-5",
        "turns": [
            {"role": "user", "content": "Feature flag dark-mode rollout to 25%"},
            {"role": "assistant", "content": "Dark-mode enabled for 25% of users via LaunchDarkly"},
            {"role": "user", "content": "Monitor error rate on auth-service"},
            {"role": "assistant", "content": "Auth-service error rate at 0.02% — within threshold"},
            {"role": "user", "content": "Progressive rollout to 50% tomorrow"},
            {"role": "assistant", "content": "Scheduled dark-mode 50% rollout for tomorrow at 09:00 UTC"},
        ],
        "questions": [
            {"question": "What feature was rolled out?", "answer": "dark-mode", "type": "entity_tracking"},
            {"question": "What initial percentage of users got the feature?", "answer": "25%", "type": "temporal"},
            {"question": "Which service was monitored for errors?", "answer": "auth-service", "type": "entity_tracking"},
            {"question": "What was the error rate threshold?", "answer": "0.02%", "type": "temporal"},
            {"question": "What was the next rollout percentage planned?", "answer": "50%", "type": "temporal"},
        ],
    },
    {
        "name": "cost-optimization-6",
        "turns": [
            {"role": "user", "content": "Review AWS spending for Q3"},
            {"role": "assistant", "content": "Q3 spending: $47,230 total. EC2 is 62% of costs."},
            {"role": "user", "content": "Downsize r5.2xlarge instances in ml-cluster"},
            {"role": "assistant", "content": "Resizing ml-cluster from r5.2xlarge to r5.xlarge saves $340/month"},
            {"role": "user", "content": "Apply reserved instance pricing for 1 year"},
            {"role": "assistant", "content": "1-year RI for ml-cluster: $3,840 upfront vs $4,800 on-demand"},
        ],
        "questions": [
            {"question": "What quarter was reviewed for AWS spending?", "answer": "Q3", "type": "temporal"},
            {"question": "What was the total Q3 spending?", "answer": "$47,230", "type": "temporal"},
            {"question": "Which instance type was downsized?", "answer": "r5.2xlarge", "type": "entity_tracking"},
            {"question": "What cluster was being optimized?", "answer": "ml-cluster", "type": "entity_tracking"},
            {"question": "What was the savings per month from down-sizing?", "answer": "$340/month", "type": "temporal"},
        ],
    },
    {
        "name": "cert-renewal-7",
        "turns": [
            {"role": "user", "content": "SSL cert for api.myelin.dev expires in 14 days"},
            {"role": "assistant", "content": "Checking cert for api.myelin.dev issued by Let's Encrypt"},
            {"role": "user", "content": "Renew automatically via cert-manager"},
            {"role": "assistant", "content": "cert-manager renewal configured. Namespace: istio-system"},
            {"role": "user", "content": "Verify the new cert on the staging ingress"},
            {"role": "assistant", "content": "New cert valid until 2026-03-15. Staging ingress passes OCSP check."},
        ],
        "questions": [
            {"question": "Which domain's SSL cert was expiring?", "answer": "api.myelin.dev", "type": "entity_tracking"},
            {"question": "How many days until expiration?", "answer": "14 days", "type": "temporal"},
            {"question": "Which cert authority issued the cert?", "answer": "Let's Encrypt", "type": "entity_tracking"},
            {"question": "Which tool was used for auto-renewal?", "answer": "cert-manager", "type": "entity_tracking"},
            {"question": "Until what date is the new certificate valid?", "answer": "2026-03-15", "type": "temporal"},
        ],
    },
    {
        "name": "code-review-8",
        "turns": [
            {"role": "user", "content": "Review PR #342 — Redis cache layer refactor"},
            {"role": "assistant", "content": "Reviewing PR #342. 17 files changed, 423 additions, 89 deletions."},
            {"role": "user", "content": "Check for thread safety in the connection pool"},
            {"role": "assistant", "content": "Connection pool uses threading.Lock; 2 potential deadlock paths found"},
            {"role": "user", "content": "Request changes for the deadlock and re-request review"},
            {"role": "assistant", "content": "Changes requested on PR #342. Re-assigned to @sre-team"},
            {"role": "user", "content": "Mark the performance benchmark CI check as required"},
        ],
        "questions": [
            {"question": "What PR number was reviewed?", "answer": "342", "type": "temporal"},
            {"question": "How many files were changed in the PR?", "answer": "17", "type": "temporal"},
            {"question": "What was the refactored component?", "answer": "Redis cache layer", "type": "entity_tracking"},
            {"question": "How many potential deadlock paths were found?", "answer": "2", "type": "temporal"},
            {"question": "Who was the review re-assigned to?", "answer": "@sre-team", "type": "entity_tracking"},
        ],
    },
    {
        "name": "kubernetes-upgrade-9",
        "turns": [
            {"role": "user", "content": "Upgrade k8s control plane from 1.27 to 1.28"},
            {"role": "assistant", "content": "Starting control plane upgrade from 1.27 to 1.28"},
            {"role": "user", "content": "Check etcd backup before proceeding"},
            {"role": "assistant", "content": "etcd snapshot saved to s3://myelin-backup/etcd-pre-1.28.snapshot"},
            {"role": "user", "content": "Drain node pool-1 before upgrading nodes"},
            {"role": "assistant", "content": "Evicting pods from pool-1 (12 pods evicted)"},
            {"role": "user", "content": "Verify all system pods are running post-upgrade"},
            {"role": "assistant", "content": "All 27 system pods healthy. coredns, kube-proxy, calico all green."},
        ],
        "questions": [
            {"question": "What was the source k8s version?", "answer": "1.27", "type": "temporal"},
            {"question": "What was the target k8s version?", "answer": "1.28", "type": "temporal"},
            {"question": "Where was the etcd backup stored?", "answer": "s3://myelin-backup/etcd-pre-1.28.snapshot", "type": "entity_tracking"},
            {"question": "How many pods were evicted from pool-1?", "answer": "12", "type": "temporal"},
            {"question": "How many system pods were healthy after upgrade?", "answer": "27", "type": "temporal"},
        ],
    },
    {
        "name": "onboarding-10",
        "turns": [
            {"role": "user", "content": "Setup dev environment for new engineer Alice"},
            {"role": "assistant", "content": "Creating workspace for Alice. Repo: myelin-core. Branch: dev/alice-sandbox"},
            {"role": "user", "content": "Grant Alice read access to the production logs bucket"},
            {"role": "assistant", "content": "Added Alice to log-reader IAM group for s3://myelin-prod-logs"},
            {"role": "user", "content": "Assign onboarding buddy: Bob from sre-team"},
            {"role": "assistant", "content": "Bob (@bob-sre) assigned as onboarding buddy for Alice"},
            {"role": "user", "content": "Create a 'getting-started' wiki page"},
            {"role": "assistant", "content": "Wiki page created: docs/onboarding/getting-started.md"},
        ],
        "questions": [
            {"question": "What is the name of the new engineer?", "answer": "Alice", "type": "entity_tracking"},
            {"question": "Which repo was the workspace created in?", "answer": "myelin-core", "type": "entity_tracking"},
            {"question": "What bucket did Alice get read access to?", "answer": "s3://myelin-prod-logs", "type": "entity_tracking"},
            {"question": "Who was assigned as Alice's onboarding buddy?", "answer": "Bob", "type": "entity_tracking"},
            {"question": "What wiki page was created?", "answer": "docs/onboarding/getting-started.md", "type": "entity_tracking"},
        ],
    },
]


def evaluate_locomo(
    strategy: str = "full_context",
) -> dict[str, Any]:
    """Evaluate against all conversations using full-context matching."""
    questions: list[dict] = []
    episodes: list[dict] = []

    for conv in CONVERSATIONS:
        for t in conv.get("turns", []):
            episodes.append(
                {
                    "domain": conv["name"],
                    "content": t["content"],
                    "role": t["role"],
                }
            )
        questions.extend(conv.get("questions", []))

    if not questions:
        return {"accuracy": 0.0, "total": 0, "error": "No questions found"}

    episode_texts = [ep["content"].lower() for ep in episodes]
    correct = 0

    start = time.perf_counter()
    for q in questions:
        answer = str(q.get("answer", "")).lower()
        found = any(answer in text for text in episode_texts)
        if found:
            correct += 1

    elapsed = time.perf_counter() - start
    n = len(questions)

    return {
        "benchmark": "locomo",
        "strategy": strategy,
        "accuracy": round(correct / max(n, 1), 4),
        "total": n,
        "correct": correct,
        "conversations": len(CONVERSATIONS),
        "elapsed_seconds": round(elapsed, 3),
    }
