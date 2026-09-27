"""Shared command-line config override validation helpers."""

import difflib

from omegaconf import DictConfig


def _context_label(context):
    return str(context or "config").strip()


def reject_append_delete_overrides(overrides, context="config"):
    label = _context_label(context)
    for override in overrides:
        override = str(override)
        if override.startswith(("++", "+", "~")):
            raise ValueError(
                f"Invalid {label} override: {override}\n"
                f"{label.capitalize()} overrides must update existing keys only; "
                "append/delete syntax is not supported."
            )


def parse_override_key(override, context="config"):
    reject_append_delete_overrides([override], context=context)

    override = str(override)
    label = _context_label(context)
    if "=" not in override:
        raise ValueError(
            f"Invalid {label} override: {override}\n"
            "Expected key=value format, for example env.humans.num_humans=10."
        )

    key = override.split("=", 1)[0].strip()
    if not key:
        raise ValueError(
            f"Invalid {label} override: {override}\n"
            "Override key cannot be empty."
        )
    return key


def _suggest_config_key(parent, prefix, missing_key):
    if not isinstance(parent, DictConfig):
        return ""

    candidates = list(parent.keys())
    matches = difflib.get_close_matches(str(missing_key), [str(item) for item in candidates], n=1)
    if not matches:
        return ""

    suggestion = f"{prefix}.{matches[0]}" if prefix else matches[0]
    return f"\nDid you mean: {suggestion}?"


def _validate_existing_leaf_override(config, key, override, context):
    label = _context_label(context)
    parts = key.split(".")
    node = config
    prefix_parts = []

    for part in parts[:-1]:
        prefix = ".".join(prefix_parts)
        if not isinstance(node, DictConfig) or part not in node:
            suggestion = _suggest_config_key(node, prefix, part)
            missing = f"{prefix}.{part}" if prefix else part
            raise ValueError(
                f"Invalid {label} override: {override}\n"
                f"Unknown config key: {missing}{suggestion}"
            )
        node = node[part]
        prefix_parts.append(part)

    leaf = parts[-1]
    prefix = ".".join(prefix_parts)
    if not isinstance(node, DictConfig) or leaf not in node:
        suggestion = _suggest_config_key(node, prefix, leaf)
        raise ValueError(
            f"Invalid {label} override: {override}\n"
            f"Unknown config key: {key}{suggestion}"
        )

    value = node[leaf]
    if isinstance(value, DictConfig):
        raise ValueError(
            f"Invalid {label} override: {override}\n"
            f"Cannot replace config section: {key}. Override an existing leaf field instead."
        )


def validate_existing_leaf_overrides(config, overrides, context="config"):
    for override in overrides:
        key = parse_override_key(override, context=context)
        _validate_existing_leaf_override(config, key, override, context)
