"""
report_writer.py — shared utility for writing step validation reports to output/.

Each pipeline script (download.py, preprocess.py, train.py) calls write_report()
at the end of its run to record a timestamped validation summary.

Reports are committed to the repo so evaluators can review them on GitHub.
They are also regenerated dynamically during docker build, so the reports in
the container always reflect the live data and freshly trained model.
"""

import os
from datetime import datetime, timezone

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")


def write_report(step: str, lines: list[str]) -> str:
    """
    Write a validation report for a pipeline step.

    Args:
        step:  filename stem, e.g. 'step_1_1_download'
        lines: list of report lines (strings)

    Returns:
        Path to the written report file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    filename = f"{step}.txt"
    path = os.path.join(OUTPUT_DIR, filename)

    header = [
        "=" * 60,
        f"Validation Report: {step}",
        f"Generated : {timestamp}",
        "=" * 60,
        "",
    ]
    footer = [
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ]

    with open(path, "w") as f:
        f.write("\n".join(header + lines + footer))

    print(f"Report written: output/{filename}")
    return path
