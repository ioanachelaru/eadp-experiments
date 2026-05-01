#!/usr/bin/env python3
"""
Parse JUnit XML test reports into per-version test execution statistics.

Reads all TEST-*.xml files from a directory (produced by Ant's <junit> task)
and sums test counts to produce a summary matching the Surefire table format
used for Calcite in the paper.

Usage:
    python parse_junit.py build/test-report 2.4.0
    python parse_junit.py build/test-report 1.4.1 --output-dir data/
"""

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

COLUMNS = ["Version", "Total", "Failed", "Error", "Skipped"]


def parse_junit_dir(report_dir):
    """
    Parse all JUnit XML files in a directory and sum test counts.

    Returns a dict with keys: tests, failures, errors, skipped.
    """
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    xml_files = sorted(Path(report_dir).glob("TEST-*.xml"))

    if not xml_files:
        print(f"Warning: no TEST-*.xml files found in {report_dir}", file=sys.stderr)
        return totals, 0

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            # <testsuite> root has attributes: tests, failures, errors, skipped
            totals["tests"] += int(root.get("tests", 0))
            totals["failures"] += int(root.get("failures", 0))
            totals["errors"] += int(root.get("errors", 0))
            totals["skipped"] += int(root.get("skipped", 0))
        except ET.ParseError as e:
            print(f"Warning: could not parse {xml_file}: {e}", file=sys.stderr)

    return totals, len(xml_files)


def main():
    parser = argparse.ArgumentParser(
        description="Parse JUnit XML reports to test execution statistics"
    )
    parser.add_argument("report_dir", help="Directory containing TEST-*.xml files")
    parser.add_argument("version", help="Ant-Ivy version (e.g., 2.4.0)")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write the output CSV (default: current directory)",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if not report_dir.is_dir():
        print(f"Error: {report_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    totals, file_count = parse_junit_dir(report_dir)
    if file_count == 0:
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"TestStats-AntIvy-{args.version}.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow({
            "Version": args.version,
            "Total": totals["tests"],
            "Failed": totals["failures"],
            "Error": totals["errors"],
            "Skipped": totals["skipped"],
        })

    print(f"Version {args.version}: {totals['tests']} tests, "
          f"{totals['failures']} failed, {totals['errors']} errors, "
          f"{totals['skipped']} skipped (from {file_count} XML files)")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
