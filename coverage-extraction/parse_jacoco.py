#!/usr/bin/env python3
"""
Parse a JaCoCo XML report into a per-file coverage CSV.

Reads the JaCoCo XML report generated after running tests with the JaCoCo
agent, and produces a CSV with per-file coverage ratios matching the format
used by the Calcite coverage CSVs in this project.

Usage:
    python parse_jacoco.py jacoco-report.xml 2.4.0
    python parse_jacoco.py jacoco-report.xml 1.4.1 --output-dir data/
"""

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

COUNTER_TYPES = ["INSTRUCTION", "BRANCH", "LINE", "COMPLEXITY", "METHOD"]
COV_COLUMNS = [f"COV_{t}" for t in COUNTER_TYPES]


def parse_counters(element):
    """Extract coverage ratios from <counter> children of an element."""
    counters = {}
    for counter in element.findall("counter"):
        ctype = counter.get("type")
        if ctype in COUNTER_TYPES:
            missed = int(counter.get("missed", 0))
            covered = int(counter.get("covered", 0))
            total = missed + covered
            counters[f"COV_{ctype}"] = covered / total if total > 0 else 0.0
    return counters


def parse_jacoco_xml(xml_path):
    """
    Parse a JaCoCo XML report into per-file coverage data.

    Returns a list of dicts with keys: filename, COV_INSTRUCTION, COV_BRANCH,
    COV_LINE, COV_COMPLEXITY, COV_METHOD.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []
    for package in root.findall(".//package"):
        pkg_name = package.get("name")  # e.g., "org/apache/ivy/core"

        for sourcefile in package.findall("sourcefile"):
            sf_name = sourcefile.get("name")  # e.g., "Ivy.java"
            filepath = f"src/java/{pkg_name}/{sf_name}"

            counters = parse_counters(sourcefile)
            row = {"filename": filepath}
            for col in COV_COLUMNS:
                row[col] = counters.get(col, 0.0)
            rows.append(row)

    return rows


def write_csv(rows, output_path):
    """Write coverage rows to CSV."""
    fieldnames = ["filename"] + COV_COLUMNS
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Parse JaCoCo XML report to per-file coverage CSV"
    )
    parser.add_argument("xml_path", help="Path to JaCoCo XML report")
    parser.add_argument("version", help="Ant-Ivy version (e.g., 2.4.0)")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write the output CSV (default: current directory)",
    )
    args = parser.parse_args()

    xml_path = Path(args.xml_path)
    if not xml_path.exists():
        print(f"Error: {xml_path} not found", file=sys.stderr)
        sys.exit(1)

    rows = parse_jacoco_xml(xml_path)
    if not rows:
        print(f"Warning: no source files found in {xml_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"Coverage-AntIvy-{args.version}-filename.csv"

    write_csv(rows, output_path)
    print(f"Wrote {len(rows)} files to {output_path}")


if __name__ == "__main__":
    main()
