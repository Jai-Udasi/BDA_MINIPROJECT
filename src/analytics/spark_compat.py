"""
PySpark and Big Data Framework Compatibility & Readiness Checker.
Inspects system prerequisites (Java, Python, Spark binaries) and provides
clear diagnostics without failing when PySpark is not yet installed.
"""
import sys
import shutil
import subprocess
import os
from pathlib import Path
from typing import Dict, Any


def check_spark_environment() -> Dict[str, Any]:
    """
    Inspects the local Windows environment for Apache Spark / PySpark prerequisites.

    Returns:
    --------
    Dict[str, Any]
        Diagnostic report covering Java, Python, and PySpark status.
    """
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # 1. Check Java JDK
    java_executable = shutil.which("java")
    java_installed = java_executable is not None
    java_version_str = "Not Found"
    java_home = os.environ.get("JAVA_HOME", "Not Configured")

    if java_installed:
        try:
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Java outputs version to stderr
            output = result.stderr if result.stderr else result.stdout
            first_line = output.splitlines()[0] if output else "Java installed (version unknown)"
            java_version_str = first_line.strip()
        except Exception as e:
            java_version_str = f"Error querying java: {str(e)}"

    # 2. Check PySpark importability
    pyspark_installed = False
    pyspark_version = "Not Installed"
    try:
        import pyspark  # type: ignore
        pyspark_installed = True
        pyspark_version = getattr(pyspark, "__version__", "Unknown")
    except ImportError:
        pyspark_installed = False
        pyspark_version = "Not Installed"

    # 3. Overall Readiness Assessment
    is_ready_for_spark = java_installed and pyspark_installed
    
    status_summary = {
        "python_version": python_version,
        "java_installed": java_installed,
        "java_executable": java_executable or "None",
        "java_version": java_version_str,
        "java_home": java_home,
        "pyspark_installed": pyspark_installed,
        "pyspark_version": pyspark_version,
        "is_ready_for_spark": is_ready_for_spark,
        "phase3_prerequisites_met": java_installed,
        "recommendation": (
            "System is ready for Phase 3! Java JDK is detected. Run 'pip install pyspark' in Phase 3."
            if java_installed and not pyspark_installed
            else ("Fully configured for PySpark execution!" if is_ready_for_spark
                  else "Please install Java JDK (Temurin/OpenJDK 17 or 21) before Phase 3.")
        )
    }
    return status_summary


if __name__ == "__main__":
    report = check_spark_environment()
    print("=== PySpark Compatibility Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
