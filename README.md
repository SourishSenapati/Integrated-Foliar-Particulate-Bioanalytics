# Integrated Foliar Particulate Bioanalytics

**Topic Focus:** _Comparison of particulate matter monitoring potential of some established plant biomonitors_

## Overview

This repository contains a comprehensive data-driven research framework analyzing the particulate matter (PM) accumulation capacities and Air Pollution Tolerance Index (APTI) thresholds of established plant biomonitors. The project integrates empirical data extracted directly from peer-reviewed literature with macro-level environmental datasets to provide a rigorous, analytical foundation for comparative biomonitoring research.

## Repository Structure

### 1. Core Research Data

- **`established_biomonitor_quantitative_data.md`**: Contains precisely verified empirical data mapping APTI scores against quantitative PM mass accumulation ($g/m^2$) for diverse plant species. All values are sourced directly from primary academic literature.
- **`verified_dois.md`**: A curated registry of Digital Object Identifiers (DOIs) for key peer-reviewed literature forming the theoretical basis of the project.
- **`research_gaps.md`**: Extracted literature gaps highlighting constraints in current biomonitoring methodologies.
- **`novel_unwritten_research_gaps.md`**: Synthesized, high-potential research vectors regarding biomonitor saturation thresholds, ultrafine particle interaction, and micro-climatic urban influences that remain unaddressed in current academic discourse.

### 2. Environmental Datasets (`PM-DATA/`)

- **`PM_Parameters_Reference.md`**: An exhaustive catalog of physical, chemical, and operational parameters for structural Particulate Matter (e.g., fractional sizes, source vectors, material hardness).
- **`pm_open_source_datasets.md`**: Index of the major open-source data platforms (EPA, NASA, Copernicus) supporting particulate measurements.
- **Automation Scripts**:
  - `download_all_csvs.py`: Automated retrieval pipeline for acquiring multi-gigabyte open-source datasets (EPA AirData, OpenAQ, geographic proxies) into the local environment for heavy statistical analysis.
  - `fetch_pm_data.py`: Base functions for EPA API connectivity and repository parsing.

## Methodology Note

All core benchmark values contained in the quantitative reference tables have been explicitly verified by the primary author through direct access to the original publisher PDFs, ensuring zero-deviation accuracy for academic benchmarking.

## License

Refer to the standard Apache 2.0 license provided in the root directory.
