import cv2
import numpy as np
import os
import datetime
from ultralytics import YOLO
from typing import Dict, Tuple
import torch
from pdfgen.pdf_generator import generate_pdf
from sms.sms_sender import SmsSender
from utils.s3_uploader import upload_to_s3
from detectors.OCR import process_image_for_ocr, OCR3
from checkers.PUCInsurance import details, convert_check

# Extra fines
PUC_FINE       = 1000
INSURANCE_FINE = 2000

# Vehicle classes in COCO/YOLO for typical vehicles
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

class Vehicle:
    def __init__(self, obj_id: int, initial_pos: Tuple[float, float], frame_idx: int):
        self.id = obj_id
        self.prev_pos = initial_pos
        self.last_frame = frame_idx
        self.speed = 0.0

    def update(self, new_pos: Tuple[float, float], frame_idx: int, fps: int, mpp: float):
        frame_delta = frame_idx - self.last_frame
        if frame_delta > 0:
            dist_px = np.linalg.norm(np.array(new_pos) - np.array(self.prev_pos))
            time_s  = frame_delta / fps
            self.speed = (dist_px * mpp) / time_s
        self.prev_pos   = new_pos
        self.last_frame = frame_idx

class VehicleSpeedDetector:
    def __init__(
        self,
        model_path: str,
        video_path: str,
        meter_per_pixel: float,
        speed_limit_mps: float,
        template_path: str,
        wkhtmltopdf_path: str,
        logo_url: str,
        twilio_sid: str,
        twilio_token: str,
        twilio_from: str,
        violator_mobile: str,
        s3_bucket: str,
        s3_region: str,
        skip_frames: int = 5,
        conf_thresh: float = 0.2,
        iou_thresh: float = 0.5,
        tracker_cfg: str = "bytetrack.yaml",
        violation_fee: int = 1500,
        ocr_instance: OCR3 = None,
    ):
        # Detection & tracking
        self.model = YOLO(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.classes = VEHICLE_CLASS_IDS

        # Video source
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Params
        self.mpp = meter_per_pixel
        self.speed_limit = speed_limit_mps
        self.skip_frames = skip_frames
        self.conf = conf_thresh
        self.iou  = iou_thresh
        self.tracker_cfg = tracker_cfg

        # PDF / SMS / S3 config
        self.template_path    = template_path
        self.wkhtmltopdf_path = wkhtmltopdf_path
        self.logo_url         = logo_url
        self.sms_sender       = SmsSender(twilio_sid, twilio_token, twilio_from)
        self.violator_mobile  = violator_mobile
        self.s3_bucket        = s3_bucket
        self.s3_region        = s3_region
        self.base_fee         = violation_fee
        self.ocr_instance     = ocr_instance

        # Internal state
        self.vehicles: Dict[int, Vehicle] = {}
        self.frame_idx = 0
        self.ticketed_ids = set()

        # Where to store frames for embedding
        self.violation_dir = os.path.join(os.getcwd(), "violations")
        os.makedirs(self.violation_dir, exist_ok=True)

    def process(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_idx += 1

            # Track vehicles
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf,
                iou=self.iou,
                tracker=self.tracker_cfg,
                classes=self.classes
            )
            for box in results[0].boxes:
                obj_id = int(box.id[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                # Skip if updated too recently
                if obj_id in self.vehicles and (self.frame_idx - self.vehicles[obj_id].last_frame) < self.skip_frames:
                    continue

                # Update or initialize
                if obj_id in self.vehicles:
                    self.vehicles[obj_id].update((cx, cy), self.frame_idx, self.fps, self.mpp)
                else:
                    self.vehicles[obj_id] = Vehicle(obj_id, (cx, cy), self.frame_idx)

                speed = self.vehicles[obj_id].speed
                # If over limit and not yet ticketed:
                if speed > self.speed_limit and obj_id not in self.ticketed_ids:
                    self.ticketed_ids.add(obj_id)
                    self._handle_speeding(frame, obj_id, speed)

            if self.frame_idx % 50 == 0:
                print(f"[{self.frame_idx}] tracked {len(self.vehicles)} vehicles; tickets issued: {len(self.ticketed_ids)}")

        self.cap.release()
        print("Done processing.")

    def _handle_speeding(self, frame, obj_id: int, speed: float):
        now_str    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        challan_no = f"S{obj_id}_{int(datetime.datetime.now().timestamp())}"

        # Save violation frame
        fname      = f"speed_{obj_id}_frame_{self.frame_idx}.jpg"
        frame_path = os.path.join(self.violation_dir, fname)
        cv2.imwrite(frame_path, frame)

        # OCR for plate
        plate = process_image_for_ocr(frame_path, ocr_instance=self.ocr_instance) or "UNKNOWN"

        # --- PUC & Insurance check ---
        # rto_info   = details(plate)
        # puc_date   = rto_info.get("PUCExpiryDate", "")
        # ins_date   = rto_info.get("InsuranceExpiryDate", "")
        puc_date   = "13-Jun-2024"  # Placeholder for testing
        ins_date   = "15-Jun-2024"  # Placeholder for testing
        puc_ok     = convert_check(puc_date) if puc_date else False
        ins_ok     = convert_check(ins_date) if ins_date else False

        extra      = 0
        desc_parts = [f"Overspeeding at {speed:.2f} m/s"]
        if not puc_ok:
            extra      += PUC_FINE
            desc_parts.append(f"Expired PUC ({puc_date})")
        if not ins_ok:
            extra      += INSURANCE_FINE
            desc_parts.append(f"Expired Insurance ({ins_date})")
        full_desc = "; ".join(desc_parts)
        total_fee = self.base_fee + extra

        # Build data dict
        data = {
            "challan_no": challan_no,
            "violation_datetime": now_str,
            "officer_name": "Traffic Inspector",
            "designation": "Traffic Police",
            "vehicle_no": plate,
            "owner_name": "",        # if known
            "owner_address": "",     # if known
            "mobile_no": self.violator_mobile,
            "payment_status": "Unpaid",
            "fee": total_fee,
            "violation_description": full_desc,
            "violation_images": [f"file:///{os.path.abspath(frame_path)}"],
            "logo_url": self.logo_url
        }
        print(data)
        # Generate PDF
        pdf_path = generate_pdf(data, self.template_path, self.wkhtmltopdf_path)
        print(f"[+] Generated speed challan PDF: {pdf_path}")

        # Upload to S3
        pdf_url = upload_to_s3(pdf_path, self.s3_bucket, self.s3_region)
        print(f"[+] Uploaded to S3: {pdf_url}")

        # Send SMS
        msg = (
            f"Download PDF: {pdf_url}"
        )
        sid = self.sms_sender.send_sms(self.violator_mobile, msg)
        # print(f"[+] SMS sent (SID: {sid})")
        print(sid)
