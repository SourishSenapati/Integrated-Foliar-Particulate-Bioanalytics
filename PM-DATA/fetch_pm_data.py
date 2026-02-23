"""Script to download open source particulate matter (PM) datasets."""

import urllib.request
import urllib.error
import urllib.parse
import json
import os

print("Starting data collection process...")

# 1. EPA AQS API Verification
USER_EMAIL = "sourishs.chem.ug@jadavpuruniversity.in"
SIGNUP_URL = f"https://aqs.epa.gov/data/api/signup?email={urllib.parse.quote(USER_EMAIL)}"
try:
    print(f"Requesting EPA AQS API access for: {USER_EMAIL}")
    req = urllib.request.Request(SIGNUP_URL, headers={
                                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    with urllib.request.urlopen(req, timeout=3) as response:
        result = json.loads(response.read().decode())
        print("EPA API Response:", result)
        if result.get("Header", []):
            print("Note: Check your email for the EPA API key.")
except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
    print(f"Failed to register EPA API: {e}")

# Ensure destination directory exists
os.makedirs("PM-DATA/sample_datasets", exist_ok=True)

# 2. Download sample global PM2.5 datasets
download_tasks = [
    {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv",
        "dest": "PM-DATA/sample_datasets/beijing_pm25.csv"
    },
    {
        "url": ("https://raw.githubusercontent.com/Azure/"
                "AzureML-Forecasting/master/Data/BeijingPM25/"
                "PRSA_data_2010.1.1-2014.12.31.csv"),
        "dest": "PM-DATA/sample_datasets/beijing_prsa_pm25.csv"
    }
]

for task in download_tasks:
    try:
        print(f"Downloading {task['dest']}...")
        req = urllib.request.Request(
            task['url'], headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            with open(task['dest'], 'wb') as f:
                f.write(response.read())
        print(f"Successfully downloaded {task['dest']}")
    except (urllib.error.URLError, OSError) as e:
        print(f"Failed to download {task['url']}: {e}")

print("Data collection completed.")
