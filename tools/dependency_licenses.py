#!/usr/bin/env python3
"""List the licenses of mBER's installed Python dependencies.

Reads requirements.txt (and protocols/setup.py's requirements) and reports each
installed package's version and license from its own metadata, so the table in
THIRD_PARTY_NOTICES.md can be checked against a real environment.

Usage:
    python tools/dependency_licenses.py          # direct dependencies
    python tools/dependency_licenses.py --all    # every installed distribution

Packages whose metadata gives no license, or a copyleft one, are flagged for review.
"""

import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COPYLEFT = re.compile(r"\b(A?GPL|LGPL|MPL|EPL|CC-BY-SA|GNU)", re.I)


def direct_requirements():
    names = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.append(re.split(r"[\s<>=!~\[@;]", line, maxsplit=1)[0])
    names += ["pyyaml"]  # protocols/setup.py
    return names


def license_of(dist):
    meta = dist.metadata
    expr = meta.get("License-Expression")
    if expr:
        return expr
    classifiers = [c.split(" :: ")[-1] for c in meta.get_all("Classifier") or []
                   if c.startswith("License ::") and c != "License :: OSI Approved"]
    if classifiers:
        return " OR ".join(classifiers)
    lic = (meta.get("License") or "").strip()
    return lic.splitlines()[0][:80] if lic and lic.upper() != "UNKNOWN" else ""


def main():
    if "--all" in sys.argv:
        dists = sorted(metadata.distributions(), key=lambda d: (d.metadata["Name"] or "").lower())
        rows = [(d.metadata["Name"], d.version, license_of(d)) for d in dists]
    else:
        rows = []
        for name in direct_requirements():
            try:
                d = metadata.distribution(name)
                rows.append((d.metadata["Name"], d.version, license_of(d)))
            except metadata.PackageNotFoundError:
                rows.append((name, "not installed", ""))
    print("| Package | Version | License | Review |")
    print("|---|---|---|---|")
    for name, version, lic in rows:
        if version == "not installed":
            flag = "not installed"
        else:
            flag = "no license metadata" if not lic else ("copyleft" if COPYLEFT.search(lic) else "")
        print(f"| {name} | {version} | {lic or '-'} | {flag} |")


if __name__ == "__main__":
    main()
