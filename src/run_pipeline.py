import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output encoding and line buffering on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from entity_resolution import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory of the student_resource workspace (default: auto-detected)",
    )
    parser.add_argument(
        "--test-dir",
        default=None,
        help="Directory containing test_source1/2/3.tsv (default: <base_dir>/dataset/test)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write output TSV files (default: <base_dir>/output)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Chunk size for streaming target files (default: 200000)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip running submission validator after pipeline completes",
    )
    args = parser.parse_args()

    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir).resolve()
    else:
        # Check current working directory, then parents of __file__
        cwd = Path.cwd().resolve()
        if (cwd / "dataset" / "test").exists():
            base_dir = cwd
        elif (cwd / "student_resource" / "dataset" / "test").exists():
            base_dir = cwd / "student_resource"
        else:
            base_dir = SRC_DIR.parents[2]  # student_resource

    test_dir = Path(args.test_dir) if args.test_dir else base_dir / "dataset" / "test"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "output"

    print("=" * 60)
    print("Business Entity Resolution Pipeline")
    print(f"Base Directory:   {base_dir}")
    print(f"Test Directory:   {test_dir}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)

    run_pipeline(
        base_dir=str(base_dir),
        test_dir=str(test_dir),
        output_dir=str(output_dir),
        chunksize=args.chunksize,
    )

    if not args.skip_validation:
        validator_script = base_dir / "utils" / "validate_submission.py"
        matching_tsv = output_dir / "matching_results.tsv"
        candidate_tsv = output_dir / "candidate_pairs.tsv"

        if validator_script.exists():
            print("\nRunning submission validator...")
            import subprocess
            cmd = [
                sys.executable,
                str(validator_script),
                "--matching", str(matching_tsv),
                "--candidate", str(candidate_tsv),
                "--test-dir", str(test_dir),
            ]
            env = dict(sys.modules["os"].environ, PYTHONIOENCODING="utf-8")
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
            if res.stdout:
                sys.stdout.write(res.stdout + "\n")
            if res.stderr:
                sys.stderr.write(res.stderr + "\n")
            if res.returncode == 0:
                print("Submission validation PASSED successfully.")
            else:
                print(f"Submission validation returned exit code {res.returncode}.")


if __name__ == "__main__":
    main()
