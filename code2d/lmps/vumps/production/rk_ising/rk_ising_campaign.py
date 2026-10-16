#!/usr/bin/env python3
"""Immutable RK-Ising campaigns with independent Z2 and cap-fix switches.

Run initialization, verification, and collection through Slurm/preempt.  No
checkpoint from another series is ever an output of this controller.  The
optional imported checkpoints are frozen byte copies, not links to live files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tomllib


PACKAGE = Path("code2d/lmps/vumps")
INPUT_ROLES = ("boundary", "direct", "z2_boundary")
INPUT_FIELDS = ("point_id", "role", "relative_path", "sha256", "source_path",
                "source_sha256", "source_config_sha256", "status")


def require_preempt():
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "preempt":
        raise RuntimeError("Run this command in a genuine Slurm/preempt allocation")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("xb") as handle:
        handle.write(payload)
    temporary.replace(path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def csv_bytes(rows, fields=None):
    fields = list(fields or dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def read_csv(path):
    payload = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    rows = list(reader)
    if not reader.fieldnames or any(None in row or None in row.values() for row in rows):
        raise ValueError("Malformed CSV: " + str(path))
    return list(reader.fieldnames), rows, hashlib.sha256(payload).hexdigest()


def series_label(use_z2, use_fix):
    return {(False, False): "no_constraint", (True, False): "Z2",
            (True, True): "Z2_fix", (False, True): "no_constraint_fix"}[(use_z2, use_fix)]


def validate_grid(fields, rows, use_z2, expected_count=None):
    required = {"task_id", "point_id", "beta", "chi", "seed"}
    if not required.issubset(fields) or not rows:
        raise ValueError("Grid requires nonempty task_id, point_id, beta, chi, seed columns")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} grid rows, found {len(rows)}")
    identities = set()
    for index, row in enumerate(rows):
        if row["task_id"] != str(index) or row["point_id"] != f"p{index:03d}":
            raise ValueError("Expected contiguous zero-based tasks and pNNN IDs")
        try:
            beta = Decimal(row["beta"])
            chi, seed = int(row["chi"]), int(row["seed"])
        except (InvalidOperation, ValueError) as error:
            raise ValueError("Invalid grid number: " + row["point_id"]) from error
        if not beta.is_finite() or beta <= 0 or chi < 2 or seed < 0 or (use_z2 and chi % 2):
            raise ValueError("Invalid beta/chi/seed or unbalanced Z2 chi: " + row["point_id"])
        identity = (beta, chi)
        if identity in identities:
            raise ValueError("Duplicate beta/chi: " + row["point_id"])
        identities.add(identity)
    groups = {}
    for memory, select in ((32, lambda chi: chi <= 20), (64, lambda chi: chi == 24),
                           (192, lambda chi: chi > 24)):
        tasks = [int(row["task_id"]) for row in rows if select(int(row["chi"]))]
        if tasks:
            groups[f"{memory}G"] = {"memory_gb": memory, "cpus_per_task": 4,
                                    "array_spec": ",".join(map(str, tasks)), "task_ids": tasks}
    return {"point_count": len(rows), "array_spec": f"0-{len(rows)-1}",
            "chis": sorted({int(row["chi"]) for row in rows}),
            "resource_groups": groups,
            "by_phase": dict(Counter(row.get("phase", "unspecified") for row in rows))}


def toml_bytes(config):
    # All campaign parameters are flat scalars/lists; reject tables rather than
    # silently emitting an invalid or ambiguous configuration.
    def encode(value):
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, (int, float)) and (not isinstance(value, float) or math.isfinite(value)):
            return repr(value)
        if isinstance(value, list):
            return "[" + ", ".join(encode(item) for item in value) + "]"
        raise ValueError("Unsupported TOML campaign value: " + repr(value))
    return ("\n".join(f"{key} = {encode(value)}" for key, value in config.items()) + "\n").encode()


def selected_sources(clean):
    selected = {path.relative_to(clean): path for path in (clean / PACKAGE / "src").rglob("*.jl")}
    selected.update({path.relative_to(clean): path for path in (clean / PACKAGE / "test").rglob("*.jl")})
    required = (
        "Project.toml", "Manifest.toml", "src/LMPSVUMPS.jl", "src/models/rk_ising_z2.jl",
        "src/algorithms/sixr_cap_selected.jl", "production/rk_ising/rk_ising_point.jl",
        "production/rk_ising/bare_strip_checks.jl", "production/rk_ising/rk_ising_campaign.py",
        "jobs/rk_ising_campaign.sh",
    )
    for name in required:
        relative = PACKAGE / name
        if not (clean / relative).is_file():
            raise ValueError("Missing campaign source: " + str(relative))
        selected[relative] = clean / relative
    for name in ("test/test_rk_ising_campaign.py", "production/rk_ising/README.md",
                 "production/rk_ising/example_config.toml", "production/rk_ising/example_grid.csv"):
        optional = PACKAGE / name
        if (clean / optional).is_file():
            selected[optional] = clean / optional
    if any(path.is_symlink() for path in selected.values()):
        raise ValueError("Source snapshot must contain regular files, not symlinks")
    return selected


def copy_stable(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(str(destination))
    # Old jobs may atomically replace their checkpoints while the new campaign
    # is initialized. Retry only our private incomplete copy; never modify the
    # old file or publish a checkpoint with uncertain provenance.
    for attempt in range(3):
        temporary = destination.with_name(destination.name + f".copy.{os.getpid()}.{attempt}")
        try:
            before = sha(source)
            with source.open("rb") as reader, temporary.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            if before == sha(temporary) == sha(source):
                if destination.exists():
                    raise FileExistsError(str(destination))
                temporary.replace(destination)
                return before
        finally:
            if temporary.exists():
                temporary.unlink()
    raise ValueError("Source kept changing during snapshot; initialization is incomplete: " + str(source))


def source_is_z2(config):
    if "use_z2" in config:
        if not isinstance(config["use_z2"], bool):
            raise ValueError("Source use_z2 must be boolean")
        return config["use_z2"]
    return config.get("boundary_symmetry") == "bosonic_Z2"


def init_campaign(args):
    clean, root = args.clean.resolve(), args.root.resolve()
    grid, config_path = args.source_grid.resolve(), args.source_config.resolve()
    reuse = args.reuse_from.resolve() if args.reuse_from else None
    if root == clean or root == Path("/") or (reuse and (root == reuse or root in reuse.parents or reuse in root.parents)):
        raise ValueError("Use an independent campaign output root")
    if root.exists() and (root / "provenance/campaign.json").is_file():
        prior, _, _, _ = verify_campaign(root)
        if (prior["use_z2"], prior["use_cap_fix"]) != (args.use_z2, args.use_cap_fix):
            raise ValueError("Existing campaign has different immutable switches")
        if prior["requested_grid_sha256"] != sha(grid) or prior["source_config_sha256"] != sha(config_path):
            raise ValueError("Existing campaign was initialized from different inputs")
        if prior.get("reuse_from", "") != (str(reuse) if reuse else ""):
            raise ValueError("Existing campaign has a different checkpoint source")
        print(json.dumps({"action": "init_already_verified", "root": str(root)}), flush=True)
        return
    if root.exists() and any(path.name != "logs" or not path.is_dir() for path in root.iterdir()):
        raise ValueError("Refusing to overwrite existing/partial campaign; only logs may preexist")
    fields, rows, grid_hash = read_csv(grid)
    stats = validate_grid(fields, rows, args.use_z2, args.expected_count)
    config = tomllib.loads(config_path.read_text())
    imported_config = None
    if reuse:
        imported_config = tomllib.loads((reuse / "config.toml").read_text())
        if source_is_z2(imported_config) != args.use_z2:
            raise ValueError("Cannot reuse boundary checkpoints across Z2/no-constraint series")
        if sha(reuse / "requested_grid.csv") != grid_hash:
            raise ValueError("Checkpoint source must have the exact same grid bytes")
    selected = selected_sources(clean)
    label = series_label(args.use_z2, args.use_cap_fix)
    config.update(schema_version=2, use_z2=args.use_z2, use_cap_fix=args.use_cap_fix,
                  series_label=label, boundary_symmetry="bosonic_Z2" if args.use_z2 else "none",
                  production_release=False, endpoint_maps_used=False,
                  initialize="matching frozen checkpoint if valid; otherwise independent seeded " + ("Z2" if args.use_z2 else "unconstrained") + " boundary",
                  source_route="VUMPS boundary (frozen reuse or fresh) -> literal bare LR/LTR -> " + ("cap-selected" if args.use_cap_fix else "legacy") + " regional six-R",
                  chi_even_rule="chi/2" if args.use_z2 else "not_applicable",
                  initialization_policy="reuse immutable matching converged boundary/direct when valid; otherwise fresh seeded boundary",
                  beta_rule="requested_grid.csv copied byte-for-byte; no point filtering")
    if args.use_cap_fix:
        config.update(cap_fix_tolerance=1e-12, cap_fix_state_tolerance=1e-10,
                      cap_fix_amplitude_tolerance=1e-10, cap_fix_window=24,
                      cap_fix_miniter=24, cap_fix_retain_unconverged=True)
    for folder in ("logs", "points", "checkpoints", "figures", "provenance", "code_snapshot"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    copy_stable(grid, root / "requested_grid.csv")
    atomic_bytes(root / "config.toml", toml_bytes(config))
    source_config_hash = copy_stable(config_path, root / "provenance/source_config.toml")
    manifest = {relative.as_posix(): copy_stable(source, root / "code_snapshot" / relative)
                for relative, source in sorted(selected.items())}
    atomic_json(root / "source_hashes.json", manifest)
    imported_hash = ""
    if reuse:
        imported_hash = copy_stable(reuse / "config.toml", root / "provenance/input_source_config.toml")
    inputs = []
    for spec in rows:
        point = spec["point_id"]
        for role in INPUT_ROLES:
            source = reuse / "checkpoints" / f"{point}_{role}.jls" if reuse else None
            relative = f"checkpoints/{point}_input_{role}.jls"
            record = dict(point_id=point, role=role, relative_path=relative, sha256="",
                          source_path=str(source) if source else "", source_sha256="",
                          source_config_sha256=imported_hash, status="missing" if reuse else "fresh")
            if source and source.is_file():
                value = copy_stable(source, root / relative)
                record.update(sha256=value, source_sha256=value, status="copied")
            inputs.append(record)
    atomic_bytes(root / "inputs.csv", csv_bytes(inputs, INPUT_FIELDS))
    provenance = dict(schema_version=2, initialized_utc=stamp(), init_job_id=os.environ["SLURM_JOB_ID"],
                      campaign_root=str(root), clean_root=str(clean), series_label=label,
                      use_z2=args.use_z2, use_cap_fix=args.use_cap_fix,
                      reuse_from=str(reuse) if reuse else "", source_grid=str(grid),
                      source_config=str(config_path), source_config_sha256=source_config_hash,
                      input_source_config_sha256=imported_hash,
                      requested_grid_sha256=sha(root / "requested_grid.csv"),
                      config_sha256=sha(root / "config.toml"),
                      source_manifest_sha256=sha(root / "source_hashes.json"),
                      inputs_sha256=sha(root / "inputs.csv"), grid_statistics=stats,
                      input_status_counts=dict(Counter(row["status"] for row in inputs)),
                      snapshot_policy="Byte-preserving copies of available MPS/direct/Z2 checkpoints, independent of old six-R completion; driver verifies identity, convergence and parent hashes before reuse. Missing/unusable inputs fall back to fresh calculation.",
                      scientific_status="Numerical diagnostics only; no automatic physical-release claim")
    atomic_json(root / "provenance/campaign.json", provenance)
    atomic_json(root / "provenance/grid_statistics.json", stats)
    atomic_bytes(root / "README.md", (README.replace("@SERIES@", label)).encode())
    verify_campaign(root)
    print(json.dumps({"action": "init", "root": str(root), "series_label": label,
                      "inputs": provenance["input_status_counts"], **stats}), flush=True)


def safe_relative(name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Unsafe manifest path: " + name)
    return relative


def verify_code(root):
    """Verify the small immutable code/config inputs, without hashing all MPS."""
    provenance = json.loads((root / "provenance/campaign.json").read_text())
    for name, key in (("config.toml", "config_sha256"), ("requested_grid.csv", "requested_grid_sha256"),
                      ("source_hashes.json", "source_manifest_sha256"), ("inputs.csv", "inputs_sha256"),
                      ("provenance/source_config.toml", "source_config_sha256")):
        if sha(root / name) != provenance[key]:
            raise ValueError("Immutable input changed: " + name)
    if provenance.get("input_source_config_sha256"):
        if sha(root / "provenance/input_source_config.toml") != provenance["input_source_config_sha256"]:
            raise ValueError("Imported source configuration changed")
    config = tomllib.loads((root / "config.toml").read_text())
    for key in ("use_z2", "use_cap_fix"):
        if not isinstance(config[key], bool) or config[key] != provenance[key]:
            raise ValueError("Invalid independent algorithm switch: " + key)
    if config["series_label"] != series_label(config["use_z2"], config["use_cap_fix"]):
        raise ValueError("Series label disagrees with algorithm switches")
    manifest = json.loads((root / "source_hashes.json").read_text())
    for name, expected in manifest.items():
        relative = safe_relative(name)
        if not relative.is_relative_to(PACKAGE):
            raise ValueError("Source outside package: " + name)
        path = root / "code_snapshot" / relative
        if path.is_symlink() or sha(path) != expected:
            raise ValueError("Frozen code changed: " + name)
    return provenance, config


def verify_campaign(root):
    provenance, config = verify_code(root)
    fields, rows, _ = read_csv(root / "requested_grid.csv")
    stats = validate_grid(fields, rows, config["use_z2"], provenance["grid_statistics"]["point_count"])
    if stats != provenance["grid_statistics"]:
        raise ValueError("Grid statistics mismatch")
    _, inputs, _ = read_csv(root / "inputs.csv")
    expected = {(row["point_id"], role) for row in rows for role in INPUT_ROLES}
    observed = set()
    for item in inputs:
        identity = (item["point_id"], item["role"])
        if identity not in expected or identity in observed:
            raise ValueError("Unexpected/duplicate input manifest record")
        observed.add(identity)
        expected_name = f"checkpoints/{item['point_id']}_input_{item['role']}.jls"
        if item["relative_path"] != expected_name:
            raise ValueError("Unexpected input checkpoint name")
        path = root / expected_name
        if item["status"] == "copied":
            if path.is_symlink() or sha(path) != item["sha256"] or item["sha256"] != item["source_sha256"]:
                raise ValueError("Imported checkpoint changed: " + expected_name)
        elif item["status"] not in ("missing", "fresh") or path.exists():
            raise ValueError("Unrecorded imported checkpoint: " + expected_name)
    if observed != expected:
        raise ValueError("Input manifest does not cover every requested point/role")
    return provenance, config, fields, rows


def finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def collect_campaign(root):
    provenance, config, fields, specs = verify_campaign(root)
    summary, hashes = [], {}
    for spec in specs:
        path = root / "points" / (spec["point_id"] + ".csv")
        row = dict(spec, series_label=config["series_label"], use_z2=config["use_z2"],
                   use_cap_fix=config["use_cap_fix"], status="not_started", stage="not_started",
                   numerical_pass=False, production_release=False, source_csv="")
        if path.is_file():
            _, results, hashes[str(path)] = read_csv(path)
            if len(results) != 1:
                raise ValueError("Expected exactly one current point record: " + str(path))
            result = results[0]
            identity_ok = result.get("point_id") == spec["point_id"] and result.get("seed") == spec["seed"]
            identity_ok &= float(result.get("beta", "nan")) == float(spec["beta"])
            identity_ok &= int(result.get("chi", "-1")) == int(spec["chi"])
            if not identity_ok:
                raise ValueError("Point identity mismatch: " + str(path))
            for key, expected in (("config_sha256", provenance["config_sha256"]),
                                  ("code_sha256", provenance["source_manifest_sha256"])):
                if result.get(key) != expected:
                    raise ValueError("Point provenance mismatch: " + str(path) + " " + key)
            if result.get("series_label") != config["series_label"]:
                raise ValueError("Cross-series result detected: " + str(path))
            for key in ("use_z2", "use_cap_fix"):
                if result.get(key, "").lower() != str(config[key]).lower():
                    raise ValueError("Point switch mismatch: " + str(path) + " " + key)
            row.update(result)
            row.update(beta=spec["beta"], result_beta_text=result.get("beta", ""), source_csv=str(path))
        row["has_entropy"] = finite(row.get("tildeS"))
        passed = str(row.get("numerical_pass")).lower() == "true"
        row["quality"] = ("complete_pass" if passed else "complete_diagnostic") if row["has_entropy"] and row["stage"] == "complete" else ("nonfinal_finite_entropy" if row["has_entropy"] else "incomplete_no_entropy")
        summary.append(row)
    atomic_bytes(root / "summary.csv", csv_bytes(summary))
    report = dict(collected_utc=stamp(), job_id=os.environ["SLURM_JOB_ID"], series_label=config["series_label"],
                  point_count=len(summary), finite_entropy_count=sum(row["has_entropy"] for row in summary),
                  by_status=dict(Counter(row["status"] for row in summary)),
                  by_quality=dict(Counter(row["quality"] for row in summary)), source_hashes=hashes,
                  summary_sha256=sha(root / "summary.csv"),
                  selection_policy="All requested points, including finite diagnostic/failed values; no PASS-only, interpolation, phase or curve-shape filtering")
    atomic_json(root / "collection.json", report)
    print(json.dumps({key: report[key] for key in ("series_label", "point_count", "finite_entropy_count", "by_status", "by_quality")}), flush=True)


README = """# RK-Ising @SERIES@ campaign

The immutable `config.toml` independently controls `use_z2` (boundary MPS
constraint) and `use_cap_fix` (cap-selected final contraction). The supported
series are `no_constraint`, `Z2`, `Z2_fix`; `no_constraint_fix` is a separately
labelled fourth diagnostic combination, never silently merged with any other.
Bare/direct LR/LTR construction is used in all four combinations.

* `requested_grid.csv`: exact source grid bytes, with no omitted failed points.
  Old grid metadata describes the source request; the authoritative effective
  symmetry and contraction switches are in config and each result record.
* `points/`: readable per-point measurements, progress and audit records.
* `checkpoints/`: saved MPS, G/B, full contractions and immutable imported inputs.
* `inputs.csv`: every requested input role, source path/hash and copied/missing
  status. Imports do not require completion of an old six-R entropy calculation.
  The driver checks identity, source hashes and boundary convergence before use.
* `summary.csv`: every requested point, including finite diagnostic failures.
* `code_snapshot/`, `source_hashes.json`, `provenance/`: frozen source and inputs.
* `figures/`, `logs/`: figures and Slurm records, separate from numerical inputs.

Unavailable or invalid imports are not dropped from the grid: the driver falls
back to a fresh seeded boundary calculation. Old source campaigns are read-only.
No new output is written into another series. Numerical convergence does not
establish the physical validity of the chosen far-boundary closure. All results
retain `production_release=false` until a separate scientific review.

Run init/verify/collect and every numerical point through Slurm/preempt using
`jobs/rk_ising_campaign.sh`. Submit array tasks without a `%N` concurrency cap;
use `provenance/grid_statistics.json` for explicit chi-dependent memory groups.
Re-running init only verifies an identical completed initialization; it never
updates the frozen code, configuration, imports or source grid.
"""


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    main.add_argument("--root", type=Path, default=os.environ.get("RK_CAMPAIGN_ROOT"), required="RK_CAMPAIGN_ROOT" not in os.environ)
    sub = main.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init")
    init.add_argument("--clean", type=Path, default=os.environ.get("CLEAN_ROOT"), required="CLEAN_ROOT" not in os.environ)
    init.add_argument("--source-grid", type=Path, required=True)
    init.add_argument("--source-config", type=Path, required=True)
    init.add_argument("--reuse-from", type=Path)
    init.add_argument("--expected-count", type=int)
    symmetry = init.add_mutually_exclusive_group(required=True)
    symmetry.add_argument("--z2", dest="use_z2", action="store_true")
    symmetry.add_argument("--no-z2", dest="use_z2", action="store_false")
    fix = init.add_mutually_exclusive_group(required=True)
    fix.add_argument("--fix", dest="use_cap_fix", action="store_true")
    fix.add_argument("--no-fix", dest="use_cap_fix", action="store_false")
    sub.add_parser("verify")
    sub.add_parser("verify-code")
    sub.add_parser("collect")
    return main


def main():
    require_preempt()
    args = parser().parse_args()
    args.root = args.root.resolve()
    if args.action == "init":
        init_campaign(args)
    elif args.action == "verify":
        provenance, _, _, _ = verify_campaign(args.root)
        print(json.dumps({"action": "verify", "root": str(args.root), "series_label": provenance["series_label"],
                          "point_count": provenance["grid_statistics"]["point_count"]}), flush=True)
    elif args.action == "verify-code":
        provenance, _ = verify_code(args.root)
        print(json.dumps({"action": "verify-code", "series_label": provenance["series_label"],
                          "source_manifest_sha256": provenance["source_manifest_sha256"]}), flush=True)
    else:
        collect_campaign(args.root)


if __name__ == "__main__":
    main()
