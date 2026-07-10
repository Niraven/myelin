# LoCoMo-S: Synthetic Adversarial Reasoning Benchmark

A synthetic benchmark for evaluating temporal reasoning, cross-conversation inference, and entity tracking in conversational AI agents. Part of the Myelin v1 evaluation harness.

## Overview

- **10 conversations**: 5-7 turns each, covering real DevOps scenarios
- **50 questions**: 5 per conversation
- **3 question types**:
  - **temporal**: When did something happen? What was the sequence?
  - **entity_tracking**: Who/what was involved? Which resource?
  - **cross_conversation**: Facts span multiple turns
- **Evaluation**: Full-context substring match (theoretical upper bound)

## Conversations

| # | Name | Scenario |
|---|------|----------|
| 1 | deploy-failure-1 | Staging deploy rollback |
| 2 | db-migration-2 | Schema migration with constraint violation |
| 3 | incident-response-3 | CPU alert and memory leak |
| 4 | secret-rotation-4 | GitHub token rotation |
| 5 | feature-launch-5 | Dark-mode progressive rollout |
| 6 | cost-optimization-6 | AWS Q3 spending review |
| 7 | cert-renewal-7 | SSL certificate renewal |
| 8 | code-review-8 | PR review with thread safety findings |
| 9 | kubernetes-upgrade-9 | K8s control plane upgrade |
| 10 | onboarding-10 | New engineer workspace setup |

## Usage

```python
from benchmarks.locomo.harness import evaluate_locomo

report = evaluate_locomo()
print(report)
```

## ⚠️ Disclaimer

**This is a synthetic approximation, not the official LoCoMo dataset.** The official LoCoMo dataset (https://github.com/salesforce/LoCoMo) contains long-context conversations with complex temporal structures and multi-hop reasoning questions. This synthetic version is designed for rapid iteration within the Myelin project and should not be cited as LoCoMo scores.
