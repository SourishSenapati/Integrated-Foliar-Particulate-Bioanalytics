# Particulate Matter (PM) Parameters Reference

This file catalogs every possible physical, chemical, and operational parameter related to Particulate Matter (PM) monitoring, answering the requirement for an exhaustive data structure from size and type to source and hardness.

## 1. Physical Parameters

- **Size Fractions (Aerodynamic Diameter):**
  - **TSP (Total Suspended Particulate):** Typically up to 100 µm.
  - **PM10 (Coarse):** $\leq$ 10 µm (Inhalable into upper respiratory tract).
  - **PM2.5 (Fine):** $\leq$ 2.5 µm (Respirable, reaches alveoli).
  - **PM1.0 (Very Fine):** $\leq$ 1.0 µm.
  - **Ultrafine Particles (UFP):** $\leq$ 0.1 µm.
- **Shape / Morphology:**
  - Spherical (e.g., fly ash, liquid droplets).
  - Irregular/Angular (e.g., mineral dust, soil, mechanically fractured matter).
  - Fibrous (e.g., asbestos, some biological materials).
  - Agglomerates/Aggregates (e.g., soot/black carbon).
- **Hardness (Mohs Scale equivalent for solid minerals):**
  - _Note: PM itself doesn't have a single hardness; its hardness depends on its mineralogical makeup._
  - Quartz / Silica Dust: Hardness ~7 (highly abrasive, common in windblown dust).
  - Alumina ($Al_2O_3$): Hardness ~9 (industrial/crustal).
  - Calcite / Limestone Dust: Hardness ~3.
  - Magnetite / Iron Oxide: Hardness ~5.5-6.5.
  - Organic/Carbonaceous PM: Soft (Hardness < 2).
- **Other Physical Properties:**
  - **Surface Area:** Usually measured in $m^2/g$ (BET specific surface area); crucial for toxicant adsorption.
  - **Density:** True density vs. Bulk density.
  - **Hygroscopicity:** Ability to absorb water and grow in size.
  - **Refractive Index:** Determines light scattering/absorption (important for climate forcing).

## 2. Chemical Composition (Type)

- **Carbonaceous Aerosols:**
  - **Black Carbon (BC) / Elemental Carbon (EC):** Strongly absorbs light (soot).
  - **Organic Carbon (OC):** Primary (directly emitted) or Secondary (formed via photochemical reactions).
- **Inorganic Ions (Secondary Aerosols):**
  - Sulfate ($SO_4^{2-}$), Nitrate ($NO_3^{-}$), Ammonium ($NH_4^{+}$).
- **Crustal Materials / Mineral Dust:**
  - Silicon (Si), Aluminum (Al), Iron (Fe), Calcium (Ca), Potassium (K).
- **Trace Metals & Heavy Metals:**
  - Lead (Pb), Cadmium (Cd), Nickel (Ni), Chromium (Cr), Vanadium (V), Zinc (Zn).
- **Biological Components (Bioaerosols):**
  - Pollen, fungal spores, bacteria, endotoxins.
- **Polycyclic Aromatic Hydrocarbons (PAHs):** Toxic organic compounds bound to PM (e.g., Benzo[a]pyrene).

## 3. Emission Sources

- **Anthropogenic (Human-made):**
  - **Combustion (Mobile):** Vehicle exhaust (diesel and gasoline engines).
  - **Non-exhaust Traffic:** Tire wear, brake wear (high in Cu, Sb, Ba), road dust resuspension.
  - **Industrial:** Power plants (coal fly ash), cement manufacturing, metal smelting.
  - **Residential:** Wood burning (biomass combustion), household cooking.
  - **Agricultural:** Fertilizers (releases ammonia forming secondary PM), burning agricultural residue.
- **Natural:**
  - **Aeolian (Windblown):** Desert dust storms.
  - **Marine:** Sea salt spray (rich in Na, Cl).
  - **Volcanic:** Ash and $SO_2$ gas (which forms sulfates).
  - **Wildfires:** Massive sources of OC and BC.
  - **Biogenic SOA:** Secondary aerosols formed from natural volatile organic compounds (e.g., isoprene, terpenes from trees).

## 4. Derived & Analytical Parameters (Data Models)

- **Source Apportionment Factors:** Extracted using Positive Matrix Factorization (PMF) or Chemical Mass Balance (CMB) to attribute mass concentration to specific sources.
- **Air Pollution Tolerance Index (APTI):** Specific to the biomonitoring of plants interacting with PM. Parameters include:
  - Leaf Extract pH.
  - Relative Water Content (RWC).
  - Ascorbic Acid content.
  - Total Chlorophyll.
- **Air Quality Index (AQI):** Based on the mass concentration of PM2.5/PM10 mapped to health-risk categories.
