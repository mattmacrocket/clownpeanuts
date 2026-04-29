"""Fake LLM endpoint service emulator (HueyDeweyLouie persona engine).

M0 ships a skeleton that exposes an OpenAI-compatible /v1/chat/completions
endpoint and echoes the last user message back. Subsequent milestones add
inference, classifier, trap layer, and persona-pack loading.

Spec: hueydeweylouie/docs/HUEYDEWEYLOUIE-SPEC.md
"""
