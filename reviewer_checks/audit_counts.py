"""Reconcile first-stage row flags and final hourly target decisions from notebooks 01 and 02."""
import argparse
from pathlib import Path

import pandas as pd


def reconcile(project: Path, output: Path) -> None:
    processed = project / "data" / "processed"
    original = pd.read_csv(processed / "pgcb_all_rows_quality_flagged.csv",
                           usecols=["source_row_id", "datetime", "quality_status",
                                    "severe_physical_inconsistency"])
    hourly = pd.read_csv(processed / "pgcb_regular_hours_advanced_audit.csv",
                         usecols=["source_row_id", "advanced_quality_status", "final_target_usable"])
    # An outer join retains the special peak and other non-hour records in the first-stage totals.
    rows = original.merge(hourly, on="source_row_id", how="outer", validate="one_to_one", indicator=True)
    if (rows["_merge"] == "right_only").any():
        raise ValueError("Hourly audit contains records absent from the source audit.")
    rows["hourly_decision"] = rows["advanced_quality_status"].fillna("OUTSIDE_HOURLY_AUDIT")
    table = pd.crosstab(rows["quality_status"], rows["hourly_decision"], margins=True, margins_name="TOTAL")
    withheld = rows["hourly_decision"].isin(["REVIEW", "EXCLUDE_TARGET"])
    retained = rows["final_target_usable"].eq(True)
    regular = rows["_merge"].eq("both")
    if not (retained | withheld).equals(regular) or (retained & withheld).any():
        raise ValueError("Final target decisions do not partition the hourly records.")
    timestamps = pd.to_datetime(rows["datetime"])
    midnight = retained & timestamps.dt.hour.eq(0)
    counts = {
        "source_records": len(rows),
        "severe_all_timestamps": rows["severe_physical_inconsistency"].sum(),
        "hourly_records": regular.sum(),
        "excluded_hourly_targets": rows["hourly_decision"].eq("EXCLUDE_TARGET").sum(),
        "review_hourly_targets": rows["hourly_decision"].eq("REVIEW").sum(),
        "withheld_hourly_targets": withheld.sum(),
        "withheld_meeting_first_stage_severity": (withheld & rows["severe_physical_inconsistency"]).sum(),
        "withheld_other": (withheld & ~rows["severe_physical_inconsistency"]).sum(),
        "usable_hourly_targets": retained.sum(),
        "usable_midnight": midnight.sum(),
        "usable_01_to_23": (retained & ~midnight).sum(),
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.Series(counts, name="count").rename_axis("measure").to_csv(output / "counts.csv")
    table.to_csv(output / "decisions_by_stage.csv")
    print(pd.Series(counts).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reconcile(args.project, args.out)
