# import cv2
# import numpy as np
# import os
# from concurrent.futures import ThreadPoolExecutor
# from ultralytics import YOLO
# from .OCR import OCR3, process_image_for_ocr

import cv2
import numpy as np
import os
import datetime
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import torch
from pdfgen.pdf_generator import generate_pdf
from sms.sms_sender import SmsSender
from utils.s3_uploader import upload_to_s3
from .OCR import process_image_for_ocr
from checkers.PUCInsurance import details, convert_check

def intersection_area_ratio(boxA, boxB):
    """
    Returns the ratio of (intersection area of boxA and boxB) / (area of boxA).
    boxA, boxB: (x1, y1, x2, y2)
    """
    Ax1, Ay1, Ax2, Ay2 = map(float, boxA)
    Bx1, By1, Bx2, By2 = map(float, boxB)

    inter_x1 = max(Ax1, Bx1)
    inter_y1 = max(Ay1, By1)
    inter_x2 = min(Ax2, Bx2)
    inter_y2 = min(Ay2, By2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area_boxA = max(0.0, (Ax2 - Ax1)) * max(0.0, (Ay2 - Ay1))
    if area_boxA == 0:
        return 0.0
    return inter_area / area_boxA

def iou(box1, box2):
    Ax1, Ay1, Ax2, Ay2 = map(float, box1)
    Bx1, By1, Bx2, By2 = map(float, box2)

    inter_x1 = max(Ax1, Bx1)
    inter_y1 = max(Ay1, By1)
    inter_x2 = min(Ax2, Bx2)
    inter_y2 = min(Ay2, By2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area_box1 = max(0.0, (Ax2 - Ax1)) * max(0.0, (Ay2 - Ay1))
    area_box2 = max(0.0, (Bx2 - Bx1)) * max(0.0, (By2 - By1))
    union_area = area_box1 + area_box2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def update_red_light_history(current_boxes, history, min_consistency=2):
    new_history = []
    for curr in current_boxes:
        matched = False
        for hist in history:
            if iou(curr, hist['box']) > 0.3:
                hist['count'] += 1
                hist['box'] = curr  # update box position
                new_history.append(hist)
                matched = True
                break
        if not matched:
            new_history.append({'box': curr, 'count': 1})
    stable_detections = [d['box'] for d in new_history if d['count'] >= min_consistency]
    return new_history, stable_detections

def adaptive_thresholds(color, avg_v):
    if color == "red":
        if avg_v < 100:
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 50, 50])
            upper_red2 = np.array([179, 255, 255])
        else:
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([179, 255, 255])
        return lower_red1, upper_red1, lower_red2, upper_red2
    elif color == "green":
        if avg_v < 100:
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
        else:
            lower_green = np.array([40, 40, 40])
            upper_green = np.array([80, 255, 255])
        return lower_green, upper_green

def detect_traffic_lights_advanced(image, color="red", area_threshold=50, circularity_threshold=0.6):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_v = np.mean(hsv[:, :, 2])

    if color == "red":
        lower_red1, upper_red1, lower_red2, upper_red2 = adaptive_thresholds("red", avg_v)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == "green":
        lower_green, upper_green = adaptive_thresholds("green", avg_v)
        mask = cv2.inRange(hsv, lower_green, upper_green)
    else:
        return []

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_threshold:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < circularity_threshold:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, x + w, y + h))

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=15, minRadius=5, maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            x, y, r = c
            box = (x - r, y - r, x + r, y + r)
            boxes.append(box)
    return boxes

def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_COMPLEX,
                              scale=0.6, text_color=(255, 255, 255),
                              background_color=(0, 0, 0), border_color=(0, 0, 255),
                              thickness=2, padding=5):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + baseline + padding),
                  background_color, cv2.FILLED)
    cv2.rectangle(frame, (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + baseline + padding),
                  border_color, thickness)
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)


# Extra fines
PUC_FINE       = 1000
INSURANCE_FINE = 2000

class TrafficViolationDetector:
    def __init__(
        self,
        video_path: str,
        model_path: str,
        template_path: str,
        wkhtmltopdf_path: str,
        logo_url: str,
        twilio_sid: str,
        twilio_token: str,
        twilio_from: str,
        owner_mobile: str,
        s3_bucket: str = "traffic-violation-pdfs",
        s3_region: str = "ap-south-1",
        officer_name: str = "John Doe",
        designation: str = "Traffic Inspector",
        violation_fee: int = 2000,
        output_width: int = 1100,
        output_height: int = 700,
        vehicle_labels=None,
        min_consistency: int = 2,
        thread_workers: int = 2,
        taillight_threshold: float = 0.9,
        violation_offset: int = 180,
        top_fraction: float = 0.02,
        ocr_instance: object = None,
    ):
        # Video + YOLO model
        self.video_path       = video_path
        self.model            = YOLO(model_path)
        device="cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Using device: {device}")
        self.model.to(device)
        self.coco             = self.model.model.names

        # PDF/SMS config
        self.template_path    = template_path
        self.wkhtmltopdf_path = wkhtmltopdf_path
        self.logo_url         = logo_url
        self.sms_sender       = SmsSender(twilio_sid, twilio_token, twilio_from)
        self.owner_mobile     = owner_mobile

        # S3 config
        self.s3_bucket        = s3_bucket
        self.s3_region        = s3_region

        # Challan metadata
        self.officer_name     = officer_name
        self.designation      = designation
        self.base_fee         = violation_fee

        # Detection params
        self.output_width     = output_width
        self.output_height    = output_height
        self.vehicle_labels   = vehicle_labels or ["bicycle","car","motorcycle","bus","truck"]
        self.min_consistency  = min_consistency
        self.taillight_threshold = taillight_threshold
        self.violation_offset = violation_offset
        self.top_fraction     = top_fraction

        # State & threading
        self.red_light_history = []
        self.fixed_blue_line_y = None
        self.executor          = ThreadPoolExecutor(max_workers=thread_workers)
        self.ocr_instance       = ocr_instance

        # Where to store violation frames
        self.violation_dir = os.path.join(os.getcwd(), "violations")
        os.makedirs(self.violation_dir, exist_ok=True)

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError("Could not open video.")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame = cv2.resize(frame, (self.output_width, self.output_height))

            # 1) Vehicle detection
            vehicle_boxes = []
            for res in self.model.predict(frame, conf=0.75, classes=[1,2,3,5,7]):
                for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
                    label = self.coco[int(cls)]
                    if label in self.vehicle_labels:
                        vehicle_boxes.append((box, conf, label))

            # 2) Red‐light detection & history
            red_boxes = detect_traffic_lights_advanced(frame, color="red")
            self.red_light_history, stable_red = update_red_light_history(
                red_boxes, self.red_light_history, self.min_consistency
            )

            # 3) Draw stable red lights & set the blue line
            for rx1, ry1, rx2, ry2 in stable_red:
                if any(
                    intersection_area_ratio((rx1,ry1,rx2,ry2), tuple(map(int, vb[0]))) 
                    >= self.taillight_threshold
                    for vb in vehicle_boxes
                ):
                    continue
                if self.fixed_blue_line_y is None:
                    self.fixed_blue_line_y = ry2 + self.violation_offset

            if self.fixed_blue_line_y:
                cv2.line(frame,
                         (0, self.fixed_blue_line_y),
                         (self.output_width, self.fixed_blue_line_y),
                         (255,0,0), 2)

            # 4) Check for crossing
            violation = False
            for box, conf, label in vehicle_boxes:
                x1,y1,x2,y2 = map(int, box)
                top_thr = y1 + int(self.top_fraction*(y2-y1))
                if self.fixed_blue_line_y and y1 <= self.fixed_blue_line_y <= top_thr:
                    violation = True
                    break

            # 5) On violation: OCR → PUC/Insurance → frame → PDF → S3 → SMS
            if violation:
                frame_filename = f"frame_{frame_count}.jpg"
                frame_path     = os.path.join(self.violation_dir, frame_filename)
                self.executor.submit(cv2.imwrite, frame_path, frame)
                # a) OCR
                plate_text = process_image_for_ocr(frame_path) or "UNKNOWN"
                now_str    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                challan_no = f"{frame_count}_{int(datetime.datetime.now().timestamp())}"

                # b) Fetch RTO info
                rto_info   = details(plate_text)
                puc_date   = rto_info.get("PUCExpiryDate", "")
                ins_date   = rto_info.get("InsuranceExpiryDate", "")
                # puc_date   = "13-Jun-2024"  # Placeholder for testing
                # ins_date   = "15-Jun-2024"  # Placeholder for testing
                puc_ok     = convert_check(puc_date) if puc_date else False
                ins_ok     = convert_check(ins_date) if ins_date else False

                # c) Compute extra fines & description
                extra      = 0
                desc_parts = ["Red-light violation"]
                if not puc_ok:
                    extra     += PUC_FINE
                    desc_parts.append(f"Expired PUC ({puc_date})")
                if not ins_ok:
                    extra     += INSURANCE_FINE
                    desc_parts.append(f"Expired Insurance ({ins_date})")
                full_desc  = "; ".join(desc_parts)
                total_fee  = self.base_fee + extra

                # d) Save the frame
                # frame_filename = f"frame_{frame_count}.jpg"
                # frame_path     = os.path.join(self.violation_dir, frame_filename)
                # self.executor.submit(cv2.imwrite, frame_path, frame)

                # e) Build PDF data
                data = {
                    "challan_no": challan_no,
                    "violation_datetime": now_str,
                    "officer_name": self.officer_name,
                    "designation": self.designation,
                    "vehicle_no": plate_text,
                    "owner_name": "",      
                    "owner_address": "",   
                    "mobile_no": self.owner_mobile,
                    "payment_status": "Unpaid",
                    "fee": total_fee,
                    "violation_description": full_desc,
                    "violation_images": [f"file:///{os.path.abspath(frame_path)}"],
                    "logo_url": self.logo_url
                }

                # f) Generate PDF
                pdf_path = generate_pdf(data, self.template_path, self.wkhtmltopdf_path)
                print(f"[+] Generated challan PDF: {pdf_path}")

                # g) Upload to S3
                pdf_url = upload_to_s3(pdf_path, self.s3_bucket, self.s3_region)
                print(f"[+] Uploaded PDF to S3: {pdf_url}")

                # h) Send SMS
                msg = (
                    f"Download PDF: {pdf_url}"
                )
                sid = self.sms_sender.send_sms(self.owner_mobile, msg)
                print(f"[+] SMS sent (SID: {sid})")

                # reset to avoid duplicates
                self.fixed_blue_line_y = None

            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames.")

        cap.release()
        self.executor.shutdown(wait=True)
        print("Done.")