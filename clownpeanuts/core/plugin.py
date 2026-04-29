"""Plugin loading and emulator instantiation."""

from __future__ import annotations

import importlib
import os

from clownpeanuts.config.schema import ServiceConfig
from clownpeanuts.services.base import ServiceEmulator


class PluginError(RuntimeError):
    pass


ALLOWED_EMULATOR_MODULES = frozenset(
    {
        "clownpeanuts.services.ssh.emulator",
        "clownpeanuts.services.http.emulator",
        "clownpeanuts.services.database.redis_emulator",
        "clownpeanuts.services.database.mysql_emulator",
        "clownpeanuts.services.database.postgres_emulator",
        "clownpeanuts.services.database.mongo_emulator",
        "clownpeanuts.services.database.memcached_emulator",
        "clownpeanuts.services.dummy.emulator",
        # HueyDeweyLouie persona engine — fake LLM endpoint deception service.
        # Spec: hueydeweylouie/docs/HUEYDEWEYLOUIE-SPEC.md
        "clownpeanuts.services.vuln_llm.emulator",
    }
)


def _allowed_modules() -> set[str]:
    extra_raw = os.environ.get("CLOWNPEANUTS_EXTRA_ALLOWED_MODULES", "")
    extras = {item.strip() for item in extra_raw.split(",") if item.strip()}
    return set(ALLOWED_EMULATOR_MODULES).union(extras)


def load_emulator_type(module_path: str) -> type[ServiceEmulator]:
    if module_path not in _allowed_modules():
        raise PluginError(f"module '{module_path}' is not in the allowed module list")
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # pragma: no cover - passthrough for diagnostics
        raise PluginError(f"failed to import module '{module_path}': {exc}") from exc

    emulator_type = getattr(module, "Emulator", None)
    if emulator_type is None:
        raise PluginError(f"module '{module_path}' does not expose Emulator")
    if not issubclass(emulator_type, ServiceEmulator):
        raise PluginError(f"Emulator in '{module_path}' must subclass ServiceEmulator")
    return emulator_type


class PluginRegistry:
    def instantiate(self, service: ServiceConfig) -> ServiceEmulator:
        emulator_type = load_emulator_type(service.module)
        return emulator_type()
