"""
WRITE THE CODE FOR INSURANCE AND PUC DETECTION HERE AND PUT THE API KEY IN THE .ENV FILE, AND REMEMBER
THE LICENSE PLATES OCR ARE IN THE TEXT FILE CALLED OCR RESULTS IN THE FOLDER CALLED OCR RESULT
"""
import os
import requests
from datetime import datetime

def details(license: str):
    try:
        url = "https://rto-vehicle-information-india.p.rapidapi.com/getVehicleInfo"
        payload = {
            "vehicle_no": license,
            "consent": "Y",
            "consent_text": "I hereby give my consent for Eccentric Labs API to fetch my information"
        }
        headers = {
            "x-rapidapi-key": os.getenv("X_RAPIDAPI_KEY"),
            "x-rapidapi-host": os.getenv("X_RAPIDAPI_HOST"),
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"Failed to fetch RTO details: {e}")

def convert_check(date_str: str) -> bool:
    """
    date_str in format 'DD-MMM-YYYY' (e.g. '05-Jun-2024').
    Returns True if expiry_date >= today.
    """
    today = datetime.today()
    expiry = datetime.strptime(date_str, "%d-%b-%Y")
    return expiry >= today
