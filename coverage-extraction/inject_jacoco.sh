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

# Step 1: Resolve dependencies
# Strategy depends on the Ant-Ivy version:
# - v1.4.1 uses pre-Apache Maven coordinates (org=apache, org=jcraft) that
#   don't exist on Maven Central, so we manually download the correct JARs.
# - v2.0+ uses standard coordinates but old Ivy HTTP resolvers that no longer
#   work, so we use a modern Ivy 2.5.2 to resolve.
cd "${ANT_IVY_DIR}"
mkdir -p lib

# Detect v1.4.1 by its namespace (jayasoft instead of apache)
if grep -q "jayasoft" ivy.xml 2>/dev/null; then
    echo "Detected pre-Apache version (v1.4.1) — downloading dependencies manually..."
    MAVEN="https://repo1.maven.org/maven2"
    curl -sfL "${MAVEN}/commons-httpclient/commons-httpclient/3.0/commons-httpclient-3.0.jar" -o lib/commons-httpclient.jar
    curl -sfL "${MAVEN}/commons-cli/commons-cli/1.0/commons-cli-1.0.jar" -o lib/commons-cli.jar
    curl -sfL "${MAVEN}/oro/oro/2.0.8/oro-2.0.8.jar" -o lib/oro.jar
    curl -sfL "${MAVEN}/com/jcraft/jsch/0.1.25/jsch-0.1.25.jar" -o lib/jsch.jar
    curl -sfL "${MAVEN}/commons-logging/commons-logging/1.0.4/commons-logging-1.0.4.jar" -o lib/commons-logging.jar
    curl -sfL "${MAVEN}/commons-codec/commons-codec/1.3/commons-codec-1.3.jar" -o lib/commons-codec.jar
    curl -sfL "${MAVEN}/junit/junit/3.8.1/junit-3.8.1.jar" -o lib/junit.jar
    echo "  Downloaded $(ls lib/*.jar | wc -l) JARs to lib/"
    # Patch build.xml: remove the init-ivy/download-ivy dependency chain.
    # The resolve target already has unless="no.resolve", but its dependency
    # init-ivy still runs and fails because it tries to load the jayasoft
    # Ivy taskdef from a JAR that no longer exists. Since we skip resolve,
    # we don't need Ivy tasks at all — just remove the dependency.
    sed -i 's/name="resolve" depends="init-ivy, prepare"/name="resolve" depends="prepare"/' \
        "${ANT_IVY_DIR}/build.xml"
    echo "  Patched build.xml to remove init-ivy dependency"
    # Remove VFS/WebDAV source files: they depend on a specific commons-vfs
    # snapshot (20060920) that no longer exists on Maven Central. These are
    # optional integration classes, not core Ivy — removing them allows the
    # rest of the codebase to compile and be tested.
    rm -rf "${ANT_IVY_DIR}/src/java/fr/jayasoft/ivy/repository/vfs"
    rm -f  "${ANT_IVY_DIR}/src/java/fr/jayasoft/ivy/resolver/VfsResolver.java"
    rm -rf "${ANT_IVY_DIR}/test/java/fr/jayasoft/ivy/repository/vfs"
    rm -f  "${ANT_IVY_DIR}/test/java/fr/jayasoft/ivy/resolver/VfsResolverTest.java"
    echo "  Removed VFS/WebDAV source files (incompatible commons-vfs snapshot)"
else
    echo "Downloading Ivy 2.5.2 for dependency resolution..."
    if [ ! -f "${IVY_JAR}" ]; then
        curl -sL "https://repo1.maven.org/maven2/org/apache/ivy/ivy/2.5.2/ivy-2.5.2.jar" -o "${IVY_JAR}"
    fi

    echo "Resolving dependencies with Ivy 2.5.2..."
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
fi

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
ANT_FLAGS="-Dno.resolve=true"
# v1.4.1 needs -Doffline=true to skip downloading Ivy from the dead jayasoft.org
if grep -q "jayasoft" ivy.xml 2>/dev/null; then
    ANT_FLAGS="${ANT_FLAGS} -Doffline=true"
fi
echo "Running ant test ${ANT_FLAGS} ..."

ant test ${ANT_FLAGS} \
    || echo "WARNING: ant test exited with non-zero status (some tests may have failed)"

# Step 3.5: Copy JUnit test reports to output directory
echo "Collecting JUnit test reports..."
REPORT_FOUND=false
for dir in "build/test-report" "build/test/report" "build/reports"; do
    if [ -d "${ANT_IVY_DIR}/${dir}" ]; then
        cp -r "${ANT_IVY_DIR}/${dir}" "${OUTPUT_DIR}/test-report"
        REPORT_FOUND=true
        echo "  Copied from ${dir}: $(ls "${OUTPUT_DIR}"/test-report/TEST-*.xml 2>/dev/null | wc -l) XML files"
        break
    fi
done
if [ "${REPORT_FOUND}" = false ]; then
    echo "  WARNING: No JUnit test report directory found"
fi

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
