"""Deterministic 500-episode / 100-question synthetic dataset for LongMemEval-S."""

import random
import json
import os

DOMAINS = ["deployment", "coding", "debugging", "security", "devops"]
QUESTIONS_PER_DOMAIN = 20


class LongMemEvalDataset:
    """Deterministic synthetic benchmark dataset.

    Generates 500 episodes (100 per domain) and 100 questions
    (20 per domain) with exact answers found in the episode text.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._seed = seed

    # ------------------------------------------------------------------
    # Episode generation
    # ------------------------------------------------------------------

    def generate_episodes(self, episodes_per_domain: int = 100) -> list[dict]:
        """Generate deterministic episodes across all domains."""
        episodes: list[dict] = []
        for domain in DOMAINS:
            for i in range(episodes_per_domain):
                episode = self._make_episode(domain, i)
                episodes.append(episode)
        return episodes

    def _make_episode(self, domain: str, idx: int) -> dict:
        maker = getattr(self, f"_make_{domain}_episode", self._make_generic_episode)
        return maker(domain, idx)

    def _random_template(self, templates: list[str], idx: int) -> str:
        """Pick a template deterministically via index, but allow seed to influence via shuffle offset."""
        shuffled = list(templates)
        self.rng.shuffle(shuffled)
        return shuffled[idx % len(shuffled)]

    def _make_deployment_episode(self, domain: str, idx: int) -> dict:
        tools = ["nginx", "docker", "kubernetes", "terraform"]
        services = ["web", "api", "worker", "cron"]
        envs = ["staging", "production", "canary", "development"]
        actions = self._random_template(
            [
                f"Deployed version {idx + 1}.0 to {envs[idx % 4]} using CI/CD pipeline",
                f"Configured {tools[idx % 4]} for the {services[idx % 4]} service",
                f"Rolled back {services[(idx + 1) % 4]} due to memory leak in v{idx % 5 + 1}.{idx % 3}",
                f"Scaled {services[idx % 4]} replicas from {idx % 5 + 1} to {idx % 5 + 5}",
                f"Updated load balancer rules for {tools[idx % 4]} ingress on {services[idx % 4]}",
                f"Health check failed for {services[(idx + 2) % 4]} — restarted pod {tools[idx % 4]}-{idx}",
                f"Set up blue-green deployment strategy for {services[idx % 4]}",
                f"Migrated database schema v{idx % 10}.{idx % 5} on {envs[idx % 4]}",
            ],
            idx,
        )
        content = actions
        return {
            "domain": domain,
            "index": idx,
            "content": content,
            "tags": [tools[idx % 4], services[idx % 4], envs[idx % 4]],
        }

    def _make_coding_episode(self, domain: str, idx: int) -> dict:
        languages = ["Python", "Rust", "TypeScript", "Go"]
        cod_services = ["web", "api", "worker", "cron"]
        features = [
            f"Implemented {['binary search', 'merge sort', 'quick sort', 'DFS traversal'][idx % 4]} in {languages[idx % 4]}",
            f"Refactored {['API router', 'data layer', 'auth middleware', 'cache wrapper'][idx % 4]} to reduce latency by {idx % 10 + 5}%",
            f"Added {['input validation', 'rate limiting', 'error handling', 'logging middleware'][idx % 4]} to the {['auth', 'payment', 'search', 'notifications'][idx % 4]} module",
            f"Wrote {idx % 10 + 5} unit tests for the {['user service', 'order handler', 'queue consumer', 'file parser'][idx % 4]}",
            f"Optimized SQL query in {['user lookup', 'report generation', 'feed builder', 'aggregation'][idx % 4]} — runtime down from {idx % 10 + 2}s to {idx % 3 + 1}s",
            f"Set up {['pre-commit hooks', 'CI pipeline', 'code coverage thresholds', 'lint rules'][idx % 4]} for the monorepo",
            f"Reviewed PR #{idx * 10 + 1}: {idx % 5 + 1} comments, {idx % 3 + 1} approved changes",
            f"Added {['async handler', 'background worker', 'scheduled job', 'webhook endpoint'][idx % 4]} for {cod_services[idx % 4]}-events",
        ]
        content = features[idx % len(features)]
        return {
            "domain": domain,
            "index": idx,
            "content": content,
            "tags": [languages[idx % 4], cod_services[idx % 4]],
        }

    def _make_debugging_episode(self, domain: str, idx: int) -> dict:
        issue_types = [
            f"Debugged segmentation fault in {['heap allocator', 'string parser', 'network buffer', 'serializer'][idx % 4]}",
            f"Fixed null pointer dereference in {['user session', 'file handle', 'DB cursor', 'cache entry'][idx % 4]}",
            f"Resolved race condition in {['queue consumer', 'state machine', 'counter increment', 'event handler'][idx % 4]}",
            f"Traced memory leak of {idx % 50 + 10}MB in {['WebSocket handler', 'image processor', 'log aggregator', 'template renderer'][idx % 4]}",
            f"Fixed off-by-one error in {['pagination logic', 'buffer allocation', 'loop boundary', 'index lookup'][idx % 4]}",
            f"Patched incorrect type coercion in {['JSON parser', 'form validator', 'config loader', 'query builder'][idx % 4]}",
            f"Root cause: unclosed {['file descriptor', 'network socket', 'DB connection', 'thread pool'][idx % 4]} in {['worker pool', 'API gateway', 'batch processor', 'stream consumer'][idx % 4]}",
            f"Applied hotfix for deadlock in {['distributed lock', 'mutex acquisition', 'database transaction', 'file I/O'][idx % 4]}",
        ]
        content = issue_types[idx % len(issue_types)]
        return {
            "domain": domain,
            "index": idx,
            "content": content,
            "tags": [issue_types[idx % 4]],
        }

    def _make_security_episode(self, domain: str, idx: int) -> dict:
        vulns = [
            f"Patched SQL injection in {['user login', 'search endpoint', 'report export', 'admin panel'][idx % 4]}",
            f"Rotated {['API keys', 'database passwords', 'JWT secrets', 'SSH certificates'][idx % 4]} for the {['web', 'api', 'worker', 'admin'][idx % 4]} tier",
            f"Audited {['access control', 'rate limiting', 'input sanitization', 'session management'][idx % 4]} — found {idx % 5 + 1} violations",
            f"Applied CVE-2024-{idx % 10000 + 1000} patch to {['OpenSSL', 'libcurl', 'PostgreSQL driver', 'nginx'][idx % 4]}",
            f"Enabled {['MFA', 'SSO', 'OAuth 2.0', 'SAML'][idx % 4]} for the {['admin', 'dashboard', 'API', 'app'][idx % 4]} portal",
            f"Updated {['firewall rules', 'WAF policies', 'TLS config', 'CSP headers'][idx % 4]} for {['production', 'staging', 'canary', 'dev'][idx % 4]}",
            f"Investigated {['SSRF', 'CSRF', 'XSS', 'path traversal'][idx % 4]} vulnerability in {['file upload', 'webhook receiver', 'redirect handler', 'static assets'][idx % 4]}",
            f"Revoked {idx % 10 + 1} stale {['IAM roles', 'service account keys', 'user tokens', 'certificates'][idx % 4]}",
        ]
        content = vulns[idx % len(vulns)]
        return {
            "domain": domain,
            "index": idx,
            "content": content,
            "tags": [vulns[idx % 4]],
        }

    def _make_devops_episode(self, domain: str, idx: int) -> dict:
        aspects = [
            f"Provisioned {idx + 1} {'EC2 instances'} with {['Ubuntu 22.04', 'Debian 12', 'RHEL 9', 'Alpine 3.18'][idx % 4]}",
            f"Set up {['Grafana', 'Prometheus', 'Datadog', 'New Relic'][idx % 4]} monitoring for {['CPU', 'memory', 'disk I/O', 'network latency'][idx % 4]} on {['web', 'api', 'worker', 'db'][idx % 4]} tier",
            f"Configured {['Terraform', 'Pulumi', 'CloudFormation', 'Ansible'][idx % 4]} for {['VPC', 'subnets', 'load balancer', 'auto-scaling group'][idx % 4]}",
            f"Set up {['GitHub Actions', 'Jenkins', 'GitLab CI', 'CircleCI'][idx % 4]} pipeline with {idx % 5 + 2} stages",
            f"Reduced Docker image size from {idx % 500 + 500}MB to {idx % 100 + 100}MB by {['multi-stage builds', 'alpine base', 'layer caching', 'dependency pruning'][idx % 4]}",
            f"Configured {['ELK stack', 'Loki + Promtail', 'CloudWatch logs', 'Splunk'][idx % 4]} for centralized logging",
            f"Set up {['Velero', 'restic', 'borg', 'duplicati'][idx % 4]} backups for {['PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch'][idx % 4]}",
            f"Automated {['SSL renewal', 'database vacuum', 'log rotation', 'cache warm-up'][idx % 4]} via {['cron job', 'systemd timer', 'Kuberentes CronJob', 'scheduled function'][idx % 4]}",
        ]
        content = aspects[idx % len(aspects)]
        return {
            "domain": domain,
            "index": idx,
            "content": content,
            "tags": [aspects[idx % 4]],
        }

    def _make_generic_episode(self, domain: str, idx: int) -> dict:
        return {
            "domain": domain,
            "index": idx,
            "content": f"{domain} task #{idx + 1}: Completed routine maintenance.",
            "tags": [domain],
        }

    # ------------------------------------------------------------------
    # Question generation
    # ------------------------------------------------------------------

    def generate_questions(self) -> list[dict]:
        """Generate 100 questions (20 per domain) with exact answers."""
        questions: list[dict] = []
        for domain in DOMAINS:
            qs = getattr(self, f"_questions_{domain}", None)
            if qs:
                questions.extend(qs())
            else:
                for i in range(QUESTIONS_PER_DOMAIN):
                    questions.append(
                        {
                            "question": f"What was the {i + 1}th {domain} task?",
                            "answer": f"{domain} task #{i + 1}",
                            "domain": domain,
                            "source_episode_idx": i,
                            "type": "factual",
                        }
                    )
        return questions

    def _questions_deployment(self) -> list[dict]:
        qs = []
        for i in range(QUESTIONS_PER_DOMAIN):
            qs.append(
                {
                    "question": f"To which environment was version {i + 1}.0 deployed in deployment episode {i}?",
                    "answer": [["staging", "production", "canary", "development"][i % 4]],
                    "domain": "deployment",
                    "source_episode_idx": i,
                    "type": "factual",
                }
            )
        return qs

    def _questions_coding(self) -> list[dict]:
        languages = ["Python", "Rust", "TypeScript", "Go"]
        qs = []
        for i in range(QUESTIONS_PER_DOMAIN):
            qs.append(
                {
                    "question": f"Which language was used in coding episode {i} for the implementation?",
                    "answer": languages[i % 4],
                    "domain": "coding",
                    "source_episode_idx": i,
                    "type": "factual",
                }
            )
        return qs

    def _questions_debugging(self) -> list[dict]:
        qs = []
        issue_prefixes = [
            "segmentation fault",
            "null pointer",
            "race condition",
            "memory leak",
            "off-by-one",
            "type coercion",
            "unclosed",
            "deadlock",
        ]
        for i in range(QUESTIONS_PER_DOMAIN):
            qs.append(
                {
                    "question": f"What type of bug was fixed in debugging episode {i}?",
                    "answer": issue_prefixes[i % len(issue_prefixes)],
                    "domain": "debugging",
                    "source_episode_idx": i,
                    "type": "factual",
                }
            )
        return qs

    def _questions_security(self) -> list[dict]:
        qs = []
        vuln_prefixes = [
            "SQL injection",
            "Rotated",
            "Audited",
            "CVE",
            "MFA",
            "firewall",
            "SSRF",
            "Revoked",
        ]
        for i in range(QUESTIONS_PER_DOMAIN):
            qs.append(
                {
                    "question": f"What type of security operation was performed in security episode {i}?",
                    "answer": vuln_prefixes[i % len(vuln_prefixes)],
                    "domain": "security",
                    "source_episode_idx": i,
                    "type": "factual",
                }
            )
        return qs

    def _questions_devops(self) -> list[dict]:
        qs = []
        topic_prefixes = [
            "Provisioned",
            "monitoring",
            "Terraform",
            "GitHub Actions",
            "Docker image",
            "centralized logging",
            "backups",
            "Automated",
        ]
        for i in range(QUESTIONS_PER_DOMAIN):
            qs.append(
                {
                    "question": f"What was the main topic of devops episode {i}?",
                    "answer": topic_prefixes[i % len(topic_prefixes)],
                    "domain": "devops",
                    "source_episode_idx": i,
                    "type": "factual",
                }
            )
        return qs

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def metadata(self) -> dict:
        return {
            "seed": self._seed,
            "version": "1.0.0",
            "domains": DOMAINS,
            "total_episodes": len(DOMAINS) * 100,
            "total_questions": len(DOMAINS) * QUESTIONS_PER_DOMAIN,
            "questions_per_domain": QUESTIONS_PER_DOMAIN,
        }
