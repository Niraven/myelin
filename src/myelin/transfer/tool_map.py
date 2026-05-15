"""Tool mapping for cross-agent transfer adaptation.

Maps common tool names to alternatives across agents, and provides
type/action extraction for step-level adaptation.
"""

from __future__ import annotations

TOOL_ALIASES: dict[str, list[str]] = {
    "git": ["gh", "github-cli"],
    "docker": ["podman", "nerdctl"],
    "npm": ["yarn", "pnpm"],
    "pip": ["pip3", "conda"],
    "kubectl": ["oc", "k"],
    "psql": ["mysql", "sqlite3"],
    "aws": ["gcloud", "az"],
    "curl": ["wget", "httpie"],
    "make": ["just", "task"],
    "pytest": ["unittest", "nose"],
    "cargo": ["rustup"],
}

# Reverse mapping: alias -> canonical tool
_TOOL_REVERSE: dict[str, str] = {}
for canonical, aliases in TOOL_ALIASES.items():
    for alias in aliases:
        _TOOL_REVERSE[alias] = canonical

TOOL_TYPE_MAP: dict[str, dict[str, str]] = {
    "git push": {"type": "git", "action": "push"},
    "git pull": {"type": "git", "action": "pull"},
    "git commit": {"type": "git", "action": "commit"},
    "git clone": {"type": "git", "action": "clone"},
    "docker build": {"type": "docker", "action": "build"},
    "docker run": {"type": "docker", "action": "run"},
    "docker push": {"type": "docker", "action": "push"},
    "npm test": {"type": "npm", "action": "test"},
    "npm build": {"type": "npm", "action": "build"},
    "npm install": {"type": "npm", "action": "install"},
    "pip install": {"type": "pip", "action": "install"},
    "pip uninstall": {"type": "pip", "action": "uninstall"},
    "pytest": {"type": "pytest", "action": "test"},
    "make": {"type": "make", "action": "build"},
    "curl": {"type": "curl", "action": "fetch"},
    "wget": {"type": "wget", "action": "fetch"},
    "kubectl apply": {"type": "kubectl", "action": "apply"},
    "kubectl get": {"type": "kubectl", "action": "get"},
}


def get_canonical_tool(tool_name: str) -> str:
    """Return the canonical name for a tool (or itself if not an alias)."""
    return _TOOL_REVERSE.get(tool_name, tool_name)


def get_aliases(tool_name: str) -> list[str]:
    """Return known aliases for a canonical tool name."""
    return list(TOOL_ALIASES.get(tool_name, []))


def find_alternative_tool(required_tool: str, target_tools: set[str]) -> str | None:
    """Find the closest available tool on the target agent.

    1. Exact match
    2. Canonical/alias match (same tool family)
    3. Prefix/substring match (e.g. "git" matches "git pull")
    """
    req = required_tool.lower().strip()
    target_lower = {t.lower().strip() for t in target_tools}

    # 1. Exact match
    if req in target_lower:
        return required_tool

    # 2. Required is an alias -> check if canonical is present
    canonical = get_canonical_tool(req)
    if canonical != req and canonical in target_lower:
        return canonical

    # 3. Required is canonical -> check if any alias is present
    for alias in get_aliases(req):
        if alias in target_lower:
            return alias

    # 4. Same family: if target has another tool in same family
    family = canonical if canonical != req else req
    if family in TOOL_ALIASES:
        for alias in TOOL_ALIASES[family]:
            if alias in target_lower:
                return alias

    # 5. Prefix/substring match: required is a prefix of target tool
    for t in target_lower:
        if (
            t.startswith(req + " ")
            or req.startswith(t + " ")
            or (" " + req) in t
            or (" " + t) in req
        ):
            return t

    return None


def extract_tool_from_step(description: str) -> str | None:
    """Extract the primary tool reference from a step description."""
    import re

    desc_lower = description.lower()

    # Check full phrase matches first (e.g. "git push")
    for phrase, meta in TOOL_TYPE_MAP.items():
        if phrase in desc_lower:
            return meta["type"]

    # Then check single-word tools
    words = re.findall(r"\b[a-z]+\b", desc_lower)
    for word in words:
        if word in TOOL_ALIASES or word in _TOOL_REVERSE:
            return get_canonical_tool(word)

    return None
