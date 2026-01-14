import cv2
import os
import datetime
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch
from pdfgen.pdf_generator import generate_pdf
from sms.sms_sender import SmsSender
# from ../utils.s3_uploader import upload_to_s3
from utils.s3_uploader import upload_to_s3
from .OCR import process_image_for_ocr
from checkers.PUCInsurance import details, convert_check

# Extra fines
PUC_FINE       = 1000
INSURANCE_FINE = 2000

class HelmetDetectionDetector:
    def __init__(
        self,
        video_path: str,
        yolo_model_path: str,
        helmet_model_path: str,
        template_path: str,
        wkhtmltopdf_path: str,
        logo_url: str,
        twilio_sid: str,
        twilio_token: str,
        twilio_from: str,
        violator_mobile: str,
        s3_bucket: str                = "traffic-violation-pdfs",
        s3_region: str                = "ap-south-1",
        conf_threshold: float         = 0.4,
        vehicle_conf: float           = 0.5,
        vehicle_buffer: int           = 10,
        line_ratio: float             = 0.35,
        violation_fee: int            = 1500,
        verbose: bool                 = False,
        ocr_instance: object          = None,
    ):
        # Video + models
        self.cap             = cv2.VideoCapture(video_path)
        self.yolo            = YOLO(yolo_model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo.to(device)
        self.helmet_model    = YOLO(helmet_model_path)
        self.helmet_model.to(device)
        self.two_wheeler_cls = [3]  # COCO class for motorcycle

        # Configs
        self.conf_threshold  = conf_threshold
        self.vehicle_conf    = vehicle_conf
        self.vehicle_buffer  = vehicle_buffer
        self.verbose         = verbose

        # Dynamic line
        assert 0 < line_ratio < 1
        self.line_ratio = line_ratio
        self.line_y     = None

        # PDF / SMS / S3
        self.template_path    = template_path
        self.wkhtmltopdf_path = wkhtmltopdf_path
        self.logo_url         = logo_url
        self.sms_sender       = SmsSender(twilio_sid, twilio_token, twilio_from)
        self.violator_mobile  = violator_mobile
        self.s3_bucket        = s3_bucket
        self.s3_region        = s3_region
        self.base_fee         = violation_fee
        self.ocr_instance     = ocr_instance

        # Where to save violation frames
        self.work_dir    = os.getcwd()
        self.violation_dir = os.path.join(self.work_dir, "violations")
        os.makedirs(self.violation_dir, exist_ok=True)

    def process_video(self):
        if not self.cap.isOpened():
            raise IOError("Cannot open video stream")

        frame_idx = 0
        tickets   = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_idx += 1

            h, w = frame.shape[:2]
            if self.line_y is None:
                self.line_y = int(h * self.line_ratio)
                if self.verbose:
                    print(f"[INFO] Violation line at y={self.line_y}")
            cv2.line(frame, (0, self.line_y), (w, self.line_y), (255,0,0), 2)

            # 1) Detect two-wheelers
            results = self.yolo.predict(frame, conf=self.vehicle_conf, classes=self.two_wheeler_cls)
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                for (x1,y1,x2,y2) in boxes:
                    bottom_y = y2
                    if abs(bottom_y - self.line_y) <= self.vehicle_buffer:
                        if self.verbose:
                            print(f"[DEBUG] Bike near line at frame {frame_idx}")

                        # 2) Helmet check
                        # pad  = int((y2-y1)*0.3)
                        # y0   = max(0, y1 - pad)
                        # crop = frame[y0:y2, x1:x2]
                        hr   = self.helmet_model.predict(frame, conf=self.conf_threshold)[0]
                        confs = hr.boxes.conf.cpu().numpy()
                        cls_ids = hr.boxes.cls.cpu().numpy().astype(int)

                        if ((cls_ids == 1) & (confs >= self.conf_threshold)).any():
                            # Violation!
                            tickets += 1

                            fname      = f"helmet_{frame_idx}.jpg"
                            frame_path = os.path.join(self.violation_dir, fname)
                            cv2.imwrite(frame_path, frame)

                            # a) OCR full frame for plate
                            plate     = process_image_for_ocr(frame_path, ocr_instance=self.ocr_instance) or "UNKNOWN"
                            now_str   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            challan_no= f"H{frame_idx}_{int(datetime.datetime.now().timestamp())}"

                            # b) Check PUC & Insurance
                            rto       = details(plate)
                            puc_date  = rto.get("PUCExpiryDate", "")
                            ins_date  = rto.get("InsuranceExpiryDate", "")
                            # puc_date   = "13-Jun-2024"  # Placeholder for testing
                            # ins_date   = "15-Jun-2024"  # Placeholder for testing
                            puc_ok    = convert_check(puc_date) if puc_date else False
                            ins_ok    = convert_check(ins_date) if ins_date else False

                            extra     = 0
                            desc      = ["Rider without helmet"]
                            if not puc_ok:
                                extra   += PUC_FINE
                                desc.append(f"Expired PUC ({puc_date})")
                            if not ins_ok:
                                extra   += INSURANCE_FINE
                                desc.append(f"Expired Insurance ({ins_date})")
                            full_desc = "; ".join(desc)
                            total_fee = self.base_fee + extra

                            # c) Save the exact frame
                            # fname      = f"helmet_{frame_idx}.jpg"
                            # frame_path = os.path.join(self.violation_dir, fname)
                            # cv2.imwrite(frame_path, frame)

                            # d) Build PDF data
                            data = {
                                "challan_no": challan_no,
                                "violation_datetime": now_str,
                                "officer_name": "John Doe",
                                "designation": "Traffic Inspector",
                                "vehicle_no": plate,
                                "owner_name": "",
                                "owner_address": "",
                                "mobile_no": self.violator_mobile,
                                "payment_status": "Unpaid",
                                "fee": total_fee,
                                "violation_description": full_desc,
                                "violation_images": [f"file:///{os.path.abspath(frame_path)}"],
                                "logo_url": self.logo_url
                            }

                            # e) Generate PDF
                            pdf_path = generate_pdf(data, self.template_path, self.wkhtmltopdf_path)
                            print(f"[+] Generated helmet challan PDF: {pdf_path}")

                            # f) Upload PDF to S3
                            pdf_url = upload_to_s3(pdf_path, self.s3_bucket, self.s3_region)
                            print(f"[+] Uploaded PDF to S3: {pdf_url}")

                            # g) Send SMS
                            msg = (
                                f"Download PDF: {pdf_url}"
                            )
                            sid = self.sms_sender.send_sms(self.violator_mobile, msg)
                            print(f"[+] SMS sent (SID: {sid})")

            if frame_idx % 100 == 0 and self.verbose:
                print(f"[INFO] Processed {frame_idx} frames, tickets: {tickets}")

        self.cap.release()
        print(f"[RESULT] Total helmet violations handled: {tickets}")
        return tickets
