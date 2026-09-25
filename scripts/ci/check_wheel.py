#!/usr/bin/env python3
"""Verify an installed oligopoly wheel is self-contained.

Run this with the interpreter of a clean virtualenv that has the built wheel
installed -- NOT from an editable install, which puts the repo root on
``sys.path`` and hides exactly the defects this script looks for:

* modules importing through a ``src.`` prefix that does not ship
* shipped subpackages missing ``__init__.py``
* console scripts pointing at names the package does not export
* core modules that need the optional ``api`` extra

Pass ``--core`` when the wheel was installed without extras: only modules
outside ``sim.api`` (and the database config it uses) are imported, which
proves the simulation core works with just its base dependencies.
"""

import importlib
import pkgutil
import sys
from importlib.metadata import entry_points
from typing import Any

CONSOLE_SCRIPTS = ("oligopoly",)
# Modules that need the `api` extra
API_MODULES = ("sim.api", "sim.config", "sim.database")


def check_imports(core_only: bool = False) -> list[str]:
    """Import every module under ``sim`` and collect failures."""
    import sim

    failures: list[str] = []
    for mod in pkgutil.walk_packages(sim.__path__, prefix="sim."):
        if core_only and mod.name.startswith(API_MODULES):
            continue
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001 - report, don't mask
            failures.append(f"{mod.name}: {exc}")
    return failures


def _console_script_entry_points() -> list[Any]:
    """Return console_scripts entry points across Python versions."""
    eps: Any = entry_points()
    # Python >= 3.10 exposes select(); older versions return a plain dict.
    if hasattr(eps, "select"):
        return list(eps.select(group="console_scripts"))
    return list(eps.get("console_scripts", []))


def check_console_scripts() -> list[str]:
    """Ensure each console script resolves to a real callable."""
    failures = []
    all_eps = _console_script_entry_points()
    for name in CONSOLE_SCRIPTS:
        matches = [e for e in all_eps if e.name == name]
        if not matches:
            failures.append(f"{name}: no console_scripts entry point found")
            continue
        entry = matches[0]
        try:
            target = entry.load()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name} -> {entry.value}: {exc}")
            continue
        if not callable(target):
            failures.append(f"{name} -> {entry.value}: resolved object is not callable")
        else:
            print(f"  {name} -> {entry.value} OK")
    return failures


def main() -> int:
    core_only = "--core" in sys.argv[1:]
    scope = "core" if core_only else "every shipped"
    print(f"Importing {scope} sim.* module...")
    failures = check_imports(core_only)
    if not failures:
        print("  all imported cleanly")

    if not core_only:
        print("Verifying console scripts...")
        failures += check_console_scripts()

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  {f}")
        return 1

    print("\nWheel is self-contained.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
