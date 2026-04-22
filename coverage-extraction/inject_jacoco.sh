#!/usr/bin/env bash
#
# Build an Ant-Ivy release with JaCoCo instrumentation and generate
# a coverage XML report.
#
# Usage:
#   ./inject_jacoco.sh <ant-ivy-dir> <jacoco-lib-dir> <output-dir>
#
# Arguments:
#   ant-ivy-dir   - Path to the Ant-Ivy source checkout (at a release tag)
#   jacoco-lib-dir - Path to the JaCoCo lib/ directory (contains jacocoagent.jar, jacococli.jar)
#   output-dir    - Directory to write jacoco.exec and jacoco-report.xml
#
# The script:
#   1. Patches build.xml to inject the JaCoCo agent into the JUnit fork
#   2. Runs `ant test` (failures are tolerated — JaCoCo still records coverage)
#   3. Generates a JaCoCo XML report from the exec data

set -euo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: $0 <ant-ivy-dir> <jacoco-lib-dir> <output-dir>"
    exit 1
fi

ANT_IVY_DIR="$(cd "$1" && pwd)"
JACOCO_LIB="$(cd "$2" && pwd)"
OUTPUT_DIR="$(cd "$3" && pwd)"

EXEC_FILE="${OUTPUT_DIR}/jacoco.exec"
REPORT_XML="${OUTPUT_DIR}/jacoco-report.xml"

echo "=== JaCoCo Coverage Extraction ==="
echo "  Ant-Ivy dir:  ${ANT_IVY_DIR}"
echo "  JaCoCo lib:   ${JACOCO_LIB}"
echo "  Output dir:   ${OUTPUT_DIR}"

# Verify required files exist
if [ ! -f "${JACOCO_LIB}/jacocoagent.jar" ]; then
    echo "Error: jacocoagent.jar not found in ${JACOCO_LIB}" >&2
    exit 1
fi
if [ ! -f "${JACOCO_LIB}/jacococli.jar" ]; then
    echo "Error: jacococli.jar not found in ${JACOCO_LIB}" >&2
    exit 1
fi
if [ ! -f "${ANT_IVY_DIR}/build.xml" ]; then
    echo "Error: build.xml not found in ${ANT_IVY_DIR}" >&2
    exit 1
fi

# Step 1: Inject JaCoCo agent into build.xml
# Insert a <jvmarg> for the JaCoCo agent after the first EMMA jvmarg line.
echo "Patching build.xml to add JaCoCo agent..."
AGENT_ARG="            <jvmarg value=\"-javaagent:${JACOCO_LIB}/jacocoagent.jar=destfile=${EXEC_FILE}\"/>"

if grep -q "jacocoagent" "${ANT_IVY_DIR}/build.xml"; then
    echo "  build.xml already patched, skipping"
else
    sed -i "/<jvmarg value=\"-Demma.coverage.out.file=/a\\
${AGENT_ARG}" "${ANT_IVY_DIR}/build.xml"
    echo "  Injected JaCoCo agent jvmarg"
fi

# Step 2: Build and run tests
echo "Running ant test..."
cd "${ANT_IVY_DIR}"

# Run tests — tolerate failures since JaCoCo still records coverage
# for whatever tests did execute. Use a generous timeout.
ant test || echo "WARNING: ant test exited with non-zero status (some tests may have failed)"

# Step 3: Generate XML report
if [ ! -f "${EXEC_FILE}" ]; then
    echo "Error: ${EXEC_FILE} not found — JaCoCo agent may not have run" >&2
    exit 1
fi

echo "Generating JaCoCo XML report..."
java -jar "${JACOCO_LIB}/jacococli.jar" report "${EXEC_FILE}" \
    --classfiles "${ANT_IVY_DIR}/build/classes" \
    --sourcefiles "${ANT_IVY_DIR}/src/java" \
    --xml "${REPORT_XML}"

echo "Done. Report: ${REPORT_XML}"
echo "  Exec size: $(du -h "${EXEC_FILE}" | cut -f1)"
