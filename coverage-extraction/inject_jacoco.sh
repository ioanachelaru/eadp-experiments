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
#   1. Resolves dependencies using a modern Ivy JAR (old Ivy can't reach Maven Central over HTTPS)
#   2. Patches build.xml to inject the JaCoCo agent into the JUnit fork
#   3. Runs `ant test` with -Dno.resolve=true (failures are tolerated)
#   4. Generates a JaCoCo XML report from the exec data

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
IVY_JAR="${OUTPUT_DIR}/ivy-bootstrap.jar"

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

# Step 1: Resolve dependencies using a modern Ivy JAR
# Old Ivy versions (2.0-2.1) use HTTP for Maven Central, which no longer works.
# We download Ivy 2.5.2 and use it to resolve dependencies, then skip
# the project's own resolve step.
echo "Downloading Ivy 2.5.2 for dependency resolution..."
if [ ! -f "${IVY_JAR}" ]; then
    curl -sL "https://repo1.maven.org/maven2/org/apache/ivy/ivy/2.5.2/ivy-2.5.2.jar" -o "${IVY_JAR}"
fi

echo "Resolving dependencies with Ivy 2.5.2..."
cd "${ANT_IVY_DIR}"
mkdir -p lib

# Use Ivy CLI to resolve and retrieve dependencies into lib/
# The retrieve pattern matches what the build.xml expects: lib/[artifact].[ext]
# Resolve each configuration separately to avoid Ivy CLI parsing issues.
for conf in default test; do
    echo "  Resolving conf: ${conf}"
    java -jar "${IVY_JAR}" \
        -ivy ivy.xml \
        -retrieve "lib/[artifact].[ext]" \
        -confs "${conf}" \
        2>&1 || echo "  WARNING: conf '${conf}' failed (may not exist in this version)"
done

echo "  Dependencies in lib/:"
ls lib/*.jar 2>/dev/null || echo "  (none)"

# Step 2: Inject JaCoCo agent into build.xml
echo "Patching build.xml to add JaCoCo agent..."
AGENT_ARG="            <jvmarg value=\"-javaagent:${JACOCO_LIB}/jacocoagent.jar=destfile=${EXEC_FILE}\"/>"

if grep -q "jacocoagent" "${ANT_IVY_DIR}/build.xml"; then
    echo "  build.xml already patched for JaCoCo, skipping"
else
    sed -i "/<jvmarg value=\"-Demma.coverage.out.file=/a\\
${AGENT_ARG}" "${ANT_IVY_DIR}/build.xml"
    echo "  Injected JaCoCo agent jvmarg"
fi

# Step 3: Build and run tests (skip resolve since we already downloaded deps)
echo "Running ant test -Dno.resolve=true ..."

ant test -Dno.resolve=true \
    || echo "WARNING: ant test exited with non-zero status (some tests may have failed)"

# Step 4: Generate XML report
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
