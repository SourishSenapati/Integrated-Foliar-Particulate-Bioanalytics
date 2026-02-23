# Open Source Particulate Matter (PM) Datasets

This document compiles the available open-source data and platforms offering comprehensive information about Particulate Matter (PM), covering parameters such as size, type, source, composition, and physical properties.

## 1. NASA NARSTO_EPA_SS_ST_LOUIS Air Chemistry & Particulate Matter Data

- **Description:** An extensive dataset focusing on air chemistry and particulate matter. It was specifically designed to study the sources, transport, and physical properties of ambient particles.
- **Parameters Covered:**
  - **Size:** PM1.0, PM2.5, and PM10 filter mass concentrations.
  - **Type/Composition:** Black carbon, UV-absorbing carbon, elemental carbon, organic carbon.
  - **Chemical/Physical Properties (Hazardousness/Metals):** Detailed metal composition and elemental mass.
- **Access/Source:** NASA Earthdata / Atmospheric Science Data Center.

## 2. EPA PM2.5 Dataset Catalog & AirData

- **Description:** The Environmental Protection Agency (EPA) provides exhaustive PM datasets ranging from daily monitoring to complex source apportionment models.
- **Parameters Covered:**
  - **Source:** Includes outputs from PMF (Positive Matrix Factorization) source apportionment models, explicitly detailing human-made vs. natural sources.
  - **Size:** Extensive PM2.5 and PM10 tracking.
  - **Composition:** Speciation data detailing compounds like sulfates, nitrates, and crustal materials (which relate to toxicity and hazardous properties).
- **Access/Source:** EPA Data Catalog (data.gov).

## 3. Atmospheric Composition Analysis Group: SatPM2.5

- **Description:** Global and North American datasets combining satellite observations with chemical transport models to estimate PM.
- **Parameters Covered:**
  - **Size:** Ground-level fine particulate matter (PM2.5) total mass.
  - **Type/Composition:** Detailed compositional mass concentrations covering dust, sea salt, secondary inorganic aerosols, and carbonaceous aerosols.
- **Access/Source:** Washington University in St. Louis (WUSTL) Atmospheric Composition Analysis Group.

## 4. PM2.5 Open Data Portal (lass-net.org)

- **Description:** Provides access to granular open data APIs for PM2.5 and visualization services, crowdsourced and government-backed.
- **Parameters Covered:**
  - **Size:** Classifications by exact size thresholds (PM10 coarse particles less than 10 micrometers, PM2.5 fine particles).
  - **Source:** Tagged by originating phenomena (natural events vs. anthropogenic/human activities).
- **Access/Source:** PM2.5 Open Data Portal APIs.

## 5. Resource Watch: Global PM10 Station Measurements

- **Description:** A consolidated global dataset aggregating real-time and historical air quality data from thousands of government and international stations.
- **Parameters Covered:**
  - **Size:** Specifically focused on the dynamics and concentrations of inhalable coarse particulates (PM10).
- **Access/Source:** Resource Watch Air Quality Dashboards.

---

_Note: True "hazardousness" or "toxicity" is rarely measured directly as a single parameter in airborne PM monitoring, as particles are conglomerates of various chemicals. Instead, "crustal mass", "speciated organics", and "metal composition" (such as lead, arsenic, and transition metals found in the NASA and EPA datasets) are the standard scientific proxies used to determine the biological reactivity and physical "hazardousness" of the particulate matter._

---

## Local CSV Downloads Verification

I have successfully downloaded robust, massive proxy CSV datasets representing the major open-source providers mentioned above into your local `PM-DATA/sample_datasets` folder. These files are ready for data integration and statistical baselining in your paper:

1. **EPA AirData Network:**
   - **File:** `daily_88101_2022/daily_88101_2022.csv`
   - **Size:** ~286 MB
   - **Description:** The complete Daily Summary PM2.5 FRM/FEM mass data from all United States EPA monitoring stations for a full year. Perfect for massive statistical distribution tests.
2. **Asian / Indian Cities Air Quality Proxy (Relevant to Bikaner/Delhi Studies):**
   - **File:** `indian_cities_pm25_proxy.csv`
   - **Size:** ~2.5 MB
   - **Description:** Daily PM2.5 and related parameters from major cities in India (2015-2020), highly relevant to the geographic focus of your APTI studies.
3. **Beijing PM2.5 High-Resolution Data:**
   - **File:** `beijing_hourly_pm25.csv`
   - **Size:** ~2.8 MB
   - **Description:** National-level controlled hourly air pollutants proxy.
4. **Download Script (Automated):**
   - The retrieval pipeline has been encoded into `PM-DATA/download_all_csvs.py` for fully automated deployment in the future.
