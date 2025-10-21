#!/usr/bin/env python3
"""
Simple CLI to read a dotted key from a YAML file.

Usage:
  python scripts/yaml_get.py <yaml_path> <dotted.key>

Examples:
  python scripts/yaml_get.py eval/configs/qwen3_vl_m2sv.yaml model.model_name
  python scripts/yaml_get.py eval/configs/qwen3_vl_m2sv.yaml inference.backend
"""

from __future__ import annotations

import sys
from typing import Any

import yaml


def get_dotted(obj: Any, dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: yaml_get.py <yaml_path> <dotted.key>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    dotted = sys.argv[2]
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        value = get_dotted(data, dotted)
        if value is None:
            return 1
        if isinstance(value, (dict, list)):
            # Print YAML for complex types
            print(yaml.safe_dump(value, sort_keys=False).strip())
        else:
            print(str(value))
        return 0
    except FileNotFoundError:
        print(f"File not found: {path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


