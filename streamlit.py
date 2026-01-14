import streamlit as st
from pathlib import Path
import tempfile
import os

# Import your detector classes
from detectors.TrafficViolations import TrafficViolationDetector
from detectors.HelmetDetection import HelmetDetectionDetector
from detectors.Speeding import VehicleSpeedDetector
from detectors.OCR import OCR3

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Twilio / AWS / paths
TWILIO_SID     = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM    = os.getenv("TWILIO_PHONE_NUMBER")
VIOLATOR_MOBILE = "+918108424468"
S3_BUCKET      = os.getenv("AWS_S3_BUCKET", "traffic-violation-pdfs")
S3_REGION      = os.getenv("AWS_REGION", "ap-south-1")

BASE_DIR       = Path(__file__).parent
MODELS_DIR     = BASE_DIR / "models"
YOLO_MODEL     = MODELS_DIR / "yolo12n.pt"
HELMET_MODEL    = MODELS_DIR / "helmet.pt"
PLATE_MODEL      = MODELS_DIR / "train3_best.pt"
TEMPLATE_PATH  = BASE_DIR / "template" / "chalan_template.html"
WKHTMLTOPDF    = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
LOGO_URI       = (BASE_DIR / "static" / "logo.jpg").resolve().as_uri()

st.title("Traffic Violation Detection System")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
run_detector = st.button("Run Detection")

if uploaded_file and run_detector:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success(f"Video saved to temporary file: {tmp_path}")

    ocr_instance = OCR3(model_path=PLATE_MODEL)

    # Run all detectors or give user choice
    st.info("Running Red Light Detector...")
    red_detector = TrafficViolationDetector(
        video_path       = tmp_path,
        model_path       = str(YOLO_MODEL),
        template_path    = str(TEMPLATE_PATH),
        wkhtmltopdf_path = WKHTMLTOPDF,
        logo_url         = LOGO_URI,
        twilio_sid       = TWILIO_SID,
        twilio_token     = TWILIO_TOKEN,
        twilio_from      = TWILIO_FROM,
        owner_mobile     = VIOLATOR_MOBILE,
        s3_bucket        = S3_BUCKET,
        s3_region        = S3_REGION,
        officer_name     = "Inspector A. Kumar",
        designation      = "Traffic Police",
        violation_fee    = 2500,
        output_width     = 1100,
        output_height    = 700,
        vehicle_labels   = ["bicycle", "car", "motorcycle", "bus", "truck"],
        min_consistency  = 2,
        taillight_threshold = 0.9,
        violation_offset = 180,
        top_fraction     = 0.02,
        thread_workers   = 4,
        ocr_instance      = ocr_instance
    )
    red_detector.process_video()

    st.info("Running Helmet Detector...")
    helmet_detector = HelmetDetectionDetector(
        video_path        = tmp_path,
        yolo_model_path   = str(YOLO_MODEL),
        helmet_model_path = str(HELMET_MODEL),
        template_path     = str(TEMPLATE_PATH),
        wkhtmltopdf_path  = WKHTMLTOPDF,
        logo_url          = LOGO_URI,
        twilio_sid        = TWILIO_SID,
        twilio_token      = TWILIO_TOKEN,
        twilio_from       = TWILIO_FROM,
        violator_mobile   = VIOLATOR_MOBILE,
        s3_bucket         = S3_BUCKET,
        s3_region         = S3_REGION,
        conf_threshold    = 0.0,
        vehicle_conf      = 0.5,
        vehicle_buffer    = 10,
        line_ratio        = 0.7,
        violation_fee     = 1000,
        verbose           = True,
        ocr_instance      = ocr_instance
    )
    helmet_detector.process_video()

    st.info("Running Speed Detector...")
    speed_detector = VehicleSpeedDetector(
        model_path        = str(YOLO_MODEL),
        video_path        = tmp_path,
        meter_per_pixel   = 0.025,
        speed_limit_mps   = 13.89,
        template_path     = str(TEMPLATE_PATH),
        wkhtmltopdf_path  = WKHTMLTOPDF,
        logo_url          = LOGO_URI,
        twilio_sid        = TWILIO_SID,
        twilio_token      = TWILIO_TOKEN,
        twilio_from       = TWILIO_FROM,
        violator_mobile   = VIOLATOR_MOBILE,
        s3_bucket         = S3_BUCKET,
        s3_region         = S3_REGION,
        skip_frames       = 5,
        conf_thresh       = 0.2,
        iou_thresh        = 0.5,
        tracker_cfg       = "bytetrack.yaml",
        violation_fee     = 1500,
        ocr_instance      = ocr_instance
    )
    speed_detector.process()

    st.success("Detection completed.")

