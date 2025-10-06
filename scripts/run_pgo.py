#!/usr/bin/env python3
"""Build an optimised extension using cargo-pgo and optionally package a wheel."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from collections.abc import Sequence
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH = ROOT / "scripts" / "profile_workload.py"
DEFAULT_WORKDIR = ROOT / "target" / "pgo"
ARTIFACT_NAMES = {"libinterpn.so", "libinterpn.dylib", "interpn.dll", "interpn.pyd"}


class CommandError(RuntimeError):
    """Raised when a subprocess exits with a non-zero status."""


def run(
    cmd: Sequence[str], *, env: dict[str, str] | None = None, cwd: Path | None = None
) -> None:
    """Execute a command, echoing it before running."""
    print("+", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True, env=env, cwd=cwd)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - aids debugging
        raise CommandError(
            f"command failed with exit code {exc.returncode}: {' '.join(cmd)}"
        ) from exc


def ensure_bench_dependencies(bench_script: Path) -> None:
    """Import required Python modules before kicking off PGO."""
    required = {"numpy"}
    if bench_script.name == "bench_cpu.py":
        required.update({"scipy", "matplotlib"})
        hint_extras = "bench"
    elif bench_script.name == "pytest_workload.py":
        required.add("pytest")
        hint_extras = "test"
    else:
        hint_extras = None

    missing: list[str] = []
    for module in sorted(required):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        deps = ", ".join(missing)
        if hint_extras:
            hint = f"uv pip install '.[{hint_extras}]'"
        else:
            hint = "uv pip install <dependency>"
        raise SystemExit(
            f"Missing Python dependencies ({deps}). "
            f"Install them via `{hint}` before running PGO."
        )


def ensure_cargo_pgo() -> None:
    """Verify that cargo-pgo is available."""
    try:
        subprocess.run(
            ["cargo", "pgo", "--version"], check=True, capture_output=True, text=True
        )
    except FileNotFoundError as exc:  # pragma: no cover - trivial guard
        raise SystemExit(
            "`cargo-pgo` is required. Install it via `cargo install cargo-pgo`."
        ) from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - environment issue
        raise SystemExit(
            "Failed to execute `cargo pgo`. "
            "Ensure the tool is installed and functional."
        ) from exc


def cargo_pgo(args: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    """Run a cargo-pgo subcommand from the project root."""
    run(["cargo", "pgo", *args], env=env, cwd=ROOT)


def find_cdylib(target_dir: Path) -> Path:
    """Locate the built shared library under the given target directory."""
    for name in ARTIFACT_NAMES:
        candidates = sorted(target_dir.rglob(name))
        if candidates:
            return min(candidates, key=lambda path: len(path.parts))
    raise SystemExit(f"Could not locate compiled cdylib in {target_dir}.")


def extension_destination() -> Path:
    """Derive the expected extension filename inside the package."""
    suffix = next((s for s in EXTENSION_SUFFIXES if "abi3" in s), EXTENSION_SUFFIXES[0])
    return ROOT / "src" / "interpn" / f"interpn{suffix}"


def install_artifact(target_dir: Path) -> Path:
    """Copy the compiled library into the Python package."""
    artifact = find_cdylib(target_dir)
    destination = extension_destination()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact, destination)
    return destination


def swap_wheel_binary(wheel_path: Path, optimized_lib: Path) -> None:
    """Replace the extension module inside an existing wheel and update RECORD."""
    if not optimized_lib.exists():
        raise SystemExit(f"Optimized library not found at {optimized_lib}")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_root = Path(tmp_str)
        with ZipFile(wheel_path, "r") as archive:
            archive.extractall(tmp_root)

        package_dir = tmp_root / "interpn"
        if not package_dir.exists():
            raise SystemExit(
                "Wheel is missing the expected 'interpn' package directory"
            )

        target_rel = None
        for candidate in package_dir.rglob(optimized_lib.name):
            if candidate.name == optimized_lib.name:
                target_rel = candidate.relative_to(tmp_root)
                break

        if target_rel is None:
            raise SystemExit(
                f"Could not find {optimized_lib.name}"
                f" inside wheel {wheel_path.name}; unable to swap binary."
            )

        target_path = tmp_root / target_rel
        shutil.copy2(optimized_lib, target_path)

        dist_info_dirs = list(tmp_root.glob("*.dist-info"))
        if len(dist_info_dirs) != 1:
            raise SystemExit(
                "Expected a single dist-info directory in wheel"
                f" {wheel_path.name}, found {len(dist_info_dirs)}"
            )
        record_path = dist_info_dirs[0] / "RECORD"
        if not record_path.exists():
            raise SystemExit(
                "Wheel RECORD file missing; cannot update hash for swapped binary"
            )

        data = optimized_lib.read_bytes()
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .decode("ascii")
            .rstrip("=")
        )
        size = len(data)
        target_rel_posix = target_rel.as_posix()

        updated_lines: list[str] = []
        replaced_record = False
        for line in record_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{target_rel_posix},"):
                updated_lines.append(f"{target_rel_posix},sha256={digest},{size}")
                replaced_record = True
            else:
                updated_lines.append(line)
        if not replaced_record:
            raise SystemExit(
                f"RECORD entry for {target_rel_posix} "
                f"not found in wheel {wheel_path.name}"
            )
        record_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

        backup = wheel_path.with_suffix(wheel_path.suffix + ".bak")
        wheel_path.replace(backup)
        try:
            with ZipFile(wheel_path, "w", ZIP_DEFLATED) as archive:
                for file_path in sorted(tmp_root.rglob("*")):
                    if file_path.is_dir() or file_path.name == "__pycache__":
                        continue
                    archive.write(file_path, file_path.relative_to(tmp_root).as_posix())
        finally:
            backup.unlink(missing_ok=True)


def run_benchmark(bench_script: Path, profiles_dir: Path) -> None:
    """Execute the Python workload while directing LLVM profiles to profiles_dir."""
    env = os.environ.copy()
    env["LLVM_PROFILE_FILE"] = str(profiles_dir / "interpn-%p-%m.profraw")
    env.setdefault("MPLBACKEND", "Agg")
    if bench_script.name == "bench_cpu.py":
        env["INTERPNPY_INTERPN_ONLY"] = "1"
    run([sys.executable, str(bench_script)], env=env, cwd=ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run profile-guided optimisation for interpn."
    )
    parser.add_argument(
        "--bench",
        type=Path,
        default=DEFAULT_BENCH,
        help="Path to the benchmark workload to execute",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=DEFAULT_WORKDIR,
        help="Directory used as CARGO_TARGET_DIR for instrumented and optimised builds",
    )
    parser.add_argument(
        "--skip-final-build",
        action="store_true",
        help="Only gather profiles and skip the final optimised build",
    )
    parser.add_argument(
        "--build-wheel",
        action="store_true",
        help="Run `uv build` and swap in the optimised binary",
    )
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        default=None,
        help="Destination directory for the final wheel (defaults to dist/)",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path)


def main() -> None:
    args = parse_args()
    bench_script = resolve_path(args.bench)
    if not bench_script.exists():
        raise SystemExit(f"Benchmark script not found: {bench_script}")

    ensure_cargo_pgo()
    ensure_bench_dependencies(bench_script)

    workdir = resolve_path(args.workdir)
    profiles_dir = workdir / "pgo-profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    cargo_env = os.environ.copy()
    cargo_env["CARGO_TARGET_DIR"] = str(workdir)

    print("Cleaning previous cargo-pgo artifacts...", flush=True)
    cargo_pgo(["clean"], env=cargo_env)

    print("Building instrumented extension with cargo-pgo...", flush=True)
    cargo_pgo(["instrument", "build", "--", "--features=python"], env=cargo_env)
    instrumented_path = install_artifact(workdir)
    print(f"Instrumented library copied to {instrumented_path}", flush=True)

    print("Running benchmark workload...", flush=True)
    run_benchmark(bench_script, profiles_dir)

    if args.skip_final_build:
        print(
            f"Skipping final build. Profiles available in {profiles_dir};"
            " the instrumented library remains installed.",
            flush=True,
        )
        return

    print("Building optimised extension with cargo-pgo...", flush=True)
    cargo_pgo(["optimize", "build", "--", "--features=python"], env=cargo_env)
    optimised_path = install_artifact(workdir)

    print(
        "PGO build complete. Optimised extension installed at",
        optimised_path,
        flush=True,
    )

    if args.build_wheel:
        wheel_out_dir = (
            resolve_path(args.wheel_dir) if args.wheel_dir else (ROOT / "dist")
        )
        wheel_out_dir.mkdir(parents=True, exist_ok=True)
        existing = {path.resolve() for path in wheel_out_dir.glob("*.whl")}

        print("Building wheel with `uv build`...", flush=True)
        run(["uv", "build", "--wheel", "--out-dir", str(wheel_out_dir)], cwd=ROOT)

        wheels = [
            path
            for path in wheel_out_dir.glob("*.whl")
            if path.resolve() not in existing
        ]
        if not wheels:
            wheels = list(wheel_out_dir.glob("*.whl"))
        if not wheels:
            raise SystemExit("uv build did not produce any wheels; unable to continue")
        wheel_path = max(wheels, key=lambda path: path.stat().st_mtime)

        swap_wheel_binary(wheel_path, optimised_path)
        print(f"Wheel built at {wheel_path} with optimised binary", flush=True)


if __name__ == "__main__":
    try:
        main()
    except CommandError as error:
        raise SystemExit(str(error)) from error
