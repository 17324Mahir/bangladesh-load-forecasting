# Auditable Next-Day Electricity Demand Forecasting in Bangladesh

This repository accompanies the ICCIT 2026 paper *Auditable Next-Day Electricity Demand Forecasting in Bangladesh Using Data Auditing, Adaptive Conformal Calibration, and Locked-Test Evaluation*. The study forecasts recorded national electricity demand for each hour from 01:00 to 23:00 on the following day. Models were selected using 2023 validation data and evaluated on 709 eligible dates from January 1, 2024 through March 8, 2026.

The [original notebooks](notebooks/) contain the nine forecasting pipelines. The [reproduction guide](reviewer_checks/README.md) gives the commands for the additional experiments reported in the revised paper. To inspect the numbers directly, open the [saved result tables](reviewer_checks/results/).

## Reproduce the revised results

1. **Get the code.** In a terminal, run the commands below. All commands in the reproduction guide start in the repository root.

   ```sh
   git clone https://github.com/17324Mahir/bangladesh-load-forecasting.git
   cd bangladesh-load-forecasting
   ```

2. **Prepare the demand data.** The experiments use `pgcb_V1_raw.csv` from Shekh and Rafi’s PGCB version 1 release. Step 1 of the guide uses Python 3.13 and curl to fetch the CSV, checks its SHA-256 checksum, and rebuilds the forecasting inputs from notebooks 01–03. For a manual download, put the CSV in `data/raw/` under the guide’s run directory. [PGCB V1 dataset](https://doi.org/10.17632/vpk8spw2mm.1) · [Setup commands](reviewer_checks/README.md#1-rebuild-the-original-inputs)
3. **Run the audit and interval checks.** Step 2 reproduces the count of withheld targets, refits Ridge with audit decisions based only on observations available at each timestamp, and measures sensitivity to the calibration window and other interval settings. [Audit and uncertainty commands](reviewer_checks/README.md#2-reproduce-the-audit-and-uncertainty-results)
4. **Run the matched-input models.** Step 3 uses the same prepared data and a separate Python 3.12 environment with TensorFlow 2.20.0. It fits Ridge and runs each of four neural models with five seeds on CPU. The guide lists the pinned dependencies and the command to summarize all 20 neural fits. [Model comparison commands](reviewer_checks/README.md#3-reproduce-the-matched-input-comparator)

## Where to find the paper’s results

| Experiment | Result on the locked test period | Saved evidence |
| --- | --- | --- |
| Original point forecasts | Ridge mean absolute error (MAE) was 437.16 MW. Using the previous day’s load as the forecast gave 649.20 MW. | [Point-model results](reviewer_checks/results/conformal/point_model_reproduction.csv) |
| Models with identical inputs | Ridge reached 436.16 MW MAE. The lowest five-seed mean among the neural models was 506.98 MW for LSTM. | [Model summary](reviewer_checks/results/matched/matched_summary.csv) · [Individual runs](reviewer_checks/results/matched/matched_per_seed.csv) |
| Audit using only past observations | On the same 709 dates, Ridge MAE was 432.23 MW with the past-only audit and 437.16 MW with the original retrospective audit. | [Audit results](reviewer_checks/results/causal/audit_metrics.csv) · [Paired differences](reviewer_checks/results/causal/audit_paired_changes.csv) |
| Prediction intervals | With a fixed 90% target, the 180-day rolling calibration by demand group and hour covered 90.35% of observed hourly loads. | [Sensitivity results](reviewer_checks/results/conformal/conformal_sensitivity.csv) |

The hourly audit withheld 74 of 90,495 integer-hour targets, including 52 exclusions and 22 marked for review. The count of 69 first-stage severity flags covers all timestamps and overlaps with the withheld records. [Audit counts](reviewer_checks/results/audit_counts/counts.csv) · [Decisions by stage](reviewer_checks/results/audit_counts/decisions_by_stage.csv)

## Run the original notebooks in Google Colab

Open notebook 01 in Colab, run its cells, allow the Google Drive mount, and upload the PGCB raw CSV when prompted. Use the same Drive account for later notebooks. They read and write intermediate files under `MyDrive/Load_Forecasting_Paper/`. [Open notebook 01 in Colab](https://colab.research.google.com/github/17324Mahir/bangladesh-load-forecasting/blob/main/notebooks/01_PGCB_Data_Cleaning.ipynb)

| Notebooks | Purpose and additional inputs |
| --- | --- |
| 01–03 | Audit the PGCB records, construct the hourly target and create the chronological training, validation and test sets. |
| 04–07 | Fit the original machine-learning and neural forecasting pipelines, then construct and evaluate prediction intervals. |
| 08 | Check agreement with Bangladesh Power Development Board records. Supply the Version 2 Excel workbook from the [BPDB archive](https://doi.org/10.17632/x7r7wdb39k.1). |
| 09–10 | Run statistical and ablation checks, then assemble the original figures and tables. |
| 11–15 | Run the weather extension. Place the historical ERA5 input `bangladesh_weather_era5_2015_2026.csv` in `data/weather/raw/` under the Drive project before starting notebook 11. The study used [Open-Meteo historical weather data](https://open-meteo.com/en/docs/historical-weather-api). |

The command-line reviewer checks require only the PGCB input. The BPDB workbook and ERA5 CSV belong to the original notebook extensions.
