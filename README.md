# Integrated Foliar Particulate Bioanalytics

This repository contains code, datasets, and analysis scripts for evaluating how effectively different plant species act as biomonitors for particulate matter (PM).

Our work bridges empirical Air Pollution Tolerance Index (APTI) data with computational modeling (PINNs) to simulate PM dispersion and calculate the financial viability of biological air filters (SylvaNode).

## Repository Overview

- **`established_biomonitor_quantitative_data.md`**: Raw empirical data mapping APTI scores to PM mass accumulation ($g/m^2$). Extracted directly from primary literature.
- **`verified_dois.md`**: Master list of DOIs for the peer-reviewed papers we used to source our baseline metrics.
- **`research_gaps.md` & `novel_unwritten_research_gaps.md`**: Notes and identified constraints in current biomonitoring methods, highlighting areas like ultrafine particle capture and saturation limits.
- **`business_and_scalability.md`**: Economic modeling for spatial deployment. Includes scaling costs, verified carbon credit (VCC) revenue projections, and ESG compliance calculations.

## Code and Modeling (`PM-DATA/`)

We've built tools to pull open-source air quality data and generate high-resolution models locally.

- **`fetch_pm_data.py` / `download_all_csvs.py`**: Automated pipelines for pulling large EPA AirData and OpenAQ datasets. Be aware that running these might require significant local storage.
- **`PM_Parameters_Reference.md` / `pm_open_source_datasets.md`**: Catalogs of physical PM parameters and index of related open-source databases.
- **`SylvaNode/generate_diagrams.py`**: This script models PM dispersion aerodynamics and projects carbon credit ROIs based on the APTI benchmarks. It outputs 600 DPI charts into `PM-DATA/SylvaNode/diagrams/`.
  - To run the models locally: `python PM-DATA/SylvaNode/generate_diagrams.py`
  - _Dependencies_: `numpy`, `matplotlib`, `seaborn`

## Data Verification Note

All baseline metrics in the reference tables were manually verified against the original publisher PDFs. If you find any discrepancies, please open an issue with the DOI and page number.

## License

Apache 2.0. See the `LICENSE` file for details.
