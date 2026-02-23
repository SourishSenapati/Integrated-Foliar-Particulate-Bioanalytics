import urllib.request
import urllib.error
import zipfile
import io
import json
import csv
import os

print("Starting comprehensive PM dataset downloads...")
os.makedirs("PM-DATA/sample_datasets", exist_ok=True)

# 1. EPA AirData (Source 2 from the MD file)
epa_url = "https://aqs.epa.gov/aqsweb/airdata/daily_88101_2022.zip"
print(
    f"Downloading EPA PM2.5 Data from: {epa_url} (Skipping if already downloaded...)")
if not os.path.exists("PM-DATA/sample_datasets/daily_88101_2022.csv"):
    try:
        req = urllib.request.Request(
            epa_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                z.extractall("PM-DATA/sample_datasets/")
        print("Successfully downloaded and extracted EPA dataset (daily_88101_2022.csv).")
    except Exception as e:
        print(f"Failed to fetch EPA data: {e}")
else:
    print("EPA Dataset daily_88101_2022.csv already exists. Skipping.")

# 2. Asian / Indian Cities Air Quality Proxy (Relevant to Bikaner/Delhi research)
indian_cities_url = "https://raw.githubusercontent.com/learning-monk/datasets/master/Indian_cities_daily_pollution_2015-2020.csv"
print(f"Fetching Indian Cities PM2.5 data from: {indian_cities_url}")
try:
    req = urllib.request.Request(indian_cities_url, headers={
                                 'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as response:
        with open("PM-DATA/sample_datasets/indian_cities_pm25_proxy.csv", 'wb') as f:
            f.write(response.read())
    print("Successfully downloaded Indian Cities PM2.5 proxy dataset.")
except Exception as e:
    print(f"Failed to fetch Indian Cities data: {e}")

# 3. Beijing PM2.5 (High resolution Hourly Data)
beijing_url = "https://raw.githubusercontent.com/prince381/air-pollution/master/data/PRSA_Data_Aotizhongxin_20130301-20170228.csv"
print(f"Downloading Beijing PM2.5 dataset from: {beijing_url}")
try:
    req = urllib.request.Request(
        beijing_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as response:
        with open("PM-DATA/sample_datasets/beijing_hourly_pm25.csv", 'wb') as f:
            f.write(response.read())
    print("Successfully downloaded Beijing PRSA PM2.5 dataset.")
except Exception as e:
    print(f"Failed to download Beijing PM2.5 data: {e}")

print("Comprehensive data downloading finished. Check the PM-DATA/sample_datasets folder.")
