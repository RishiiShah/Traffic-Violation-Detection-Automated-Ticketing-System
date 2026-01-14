# import logging
# import torch
# import os
# from ultralytics import YOLO
# import base64
# import numpy as np
# import cv2
# from dotenv import load_dotenv
# # from google.colab.patches import cv2_imshow
# from groq import Groq

# logging.basicConfig(level=logging.INFO)

# class OCR3:
#     def __init__(self, **args):
#         # load_dotenv()
#         self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#         self.model = YOLO("/content/best.pt")  # Load your YOLOv8 model
#         torch.no_grad()  # Disable gradient computation to save memory

#     def convert_image_to_data_url(self, image: np.ndarray) -> str:
#         """Converts an image (NumPy array) to a data URL (base64 encoded)."""
#         success, buffer = cv2.imencode('.jpg', image)
#         if not success:
#             raise ValueError("Image encoding failed.")
#         image_bytes = buffer.tobytes()
#         image_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
#         return image_url

#     def crop_to_roi(self, img: np.ndarray, xyxy: list) -> np.ndarray:
#         """Crops the image to the Region of Interest (ROI) using the given bounding box."""
#         xmin, ymin, xmax, ymax = map(int, xyxy)
#         roi = img[ymin:ymax, xmin:xmax]
#         return roi

#     def detect_number_plate(self, frame) -> list:
#         """Detects a number plate in the frame using YOLO."""
#         # Optionally convert from BGR (cv2 default) to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         with torch.no_grad():
#             results = self.model(frame_rgb)

#         plate_details = []
#         # Assuming one image was passed so we use the first result
#         result = results[0]
#         for box in result.boxes:
#             # Instead of indexing [0], get the full bounding box
#             xyxy = box.xyxy.cpu().numpy().flatten()  # Ensures a flat array of 4 numbers
#             score = box.conf.cpu().numpy().item() if box.conf is not None else 0.0
#             cls_id = int(box.cls.cpu().numpy().item())
#             label = self.model.names[cls_id].lower()

#         return list(xyxy)

#     def OCR_Inference(self, image_url: str) -> str:
#         """Uses Groq API to extract text from the license plate image URL."""
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "read the license plate, output should just be the text"
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {"url": image_url}
#                     }
#                 ]
#             }
#         ]

#         try:
#             completion = self.client.chat.completions.create(
#                 model="llama-3.2-11b-vision-preview",
#                 messages=messages,
#                 temperature=1,
#                 max_completion_tokens=1024,
#                 top_p=1,
#                 stream=False,
#                 stop=None,
#             )
#             # Extract the text directly from the message attribute
#             output_text = completion.choices[0].message.content.strip()
#             return output_text

#         except Exception as e:
#             print(f"Error during OCR inference: {e}")
#             return None

#     def perform_ocr(self, frame) -> str:
#         """Performs OCR on a single frame."""

#         # Detect number plate
#         plate_info = self.detect_number_plate(frame)

#         # Crop the detected region
#         cropped_frame = self.crop_to_roi(frame, plate_info)

#         if cropped_frame is None or cropped_frame.size == 0:
#             logging.warning("Cropped frame is empty.")
#             return "Cropped frame is empty."

#         # Convert to data URL for OCR inference
#         image_url = self.convert_image_to_data_url(frame)
#         # Perform OCR Inference (synchronous call)
#         try:
#             ocr_result = self.OCR_Inference(image_url=image_url)
#             if ocr_result is None or ocr_result.strip() == "":
#                 logging.warning("OCR returned no result.")
#                 return "OCR returned no result."
#             return ocr_result
#         except Exception as e:
#             logging.error(f"Error during OCR inference: {e}")
#             return "OCR inference failed."

#     def batch_ocr(self, frames: list) -> list:
#         """Performs OCR for multiple frames synchronously."""
#         ocr_results = []
#         for frame in frames:
#             ocr_result = self.perform_ocr(frame)
#             ocr_results.append(ocr_result)
#         return ocr_results

import logging
import os
import base64
import re
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from dotenv import load_dotenv
from groq import Groq

logging.basicConfig(level=logging.INFO)

# Pre-compiled plate regexes
PLATE_PATTERNS = [
    re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$'),
    re.compile(r'^[0-9]{2}[A-Z]{2}[0-9]{4}[A-Z]{2}$'),
]

class OCR3:
    def __init__(self, model_path=None):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=api_key)
        self.model = YOLO(model_path)
        torch.no_grad()

        # Prepare results directory
        detectors_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(detectors_dir)
        self.ocr_results_dir = os.path.join(parent_dir, "ocr results")
        os.makedirs(self.ocr_results_dir, exist_ok=True)

    def convert_image_to_data_url(self, image: np.ndarray) -> str:
        success, buf = cv2.imencode('.jpg', image)
        if not success:
            raise RuntimeError("Image encoding failed")
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    def detect_number_plate(self, frame: np.ndarray) -> list[float] | None:
        """Return bbox [xmin, ymin, xmax, ymax] or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.model(rgb)[0]
        for box in result.boxes:
            cls_id = int(box.cls.cpu().item())
            if self.model.names[cls_id].lower() == "license_plate":
                return box.xyxy.cpu().numpy().flatten().tolist()
        return None

    def crop_to_roi(self, img: np.ndarray, xyxy: list[float]) -> np.ndarray:
        xmin, ymin, xmax, ymax = map(int, xyxy)
        return img[ymin:ymax, xmin:xmax]

    def OCR_Inference(self, image_url: str) -> str | None:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "read the license plate, output only the text. I want you to take your time think 10 times and come up with the best answer. I need the license plate number as accurate as possible. "
                    "Patterns: [A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4} or "
                    "[0-9]{2}[A-Z]{2}[0-9]{4}[A-Z]{2}."
                    "Your Output should be just the text of the number plate nothing else. JJUST GIVE ME THE TEXT AS ACCCURATE AS POSSIBLE THINK 10 times. "
                )},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }]
        try:
            resp = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                temperature=1.0,
                max_completion_tokens=256,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OCR API error: {e}")
            return None

    def save_ocr_result(self, plate: str) -> None:
        path = os.path.join(self.ocr_results_dir, "ocr_results.txt")
        if os.path.exists(path):
            with open(path, "r") as f:
                existing_plates = f.read().split(",")
            if plate in existing_plates:
                logging.info(f"'{plate}' already exists in {path}. Skipping append.")
                return
        with open(path, "a") as f:
            f.write(plate + ",")
        logging.info(f"Appended '{plate}' to {path}")

    def perform_ocr(self, frame: np.ndarray) -> str:
        """
        1) detect plate
        2) if none, bail
        3) crop, OCR, validate, save
        """
        # bbox = self.detect_number_plate(frame)
        # if bbox is None:
        #     logging.warning("No plate detected.")
        #     return "No plate detected."

        # roi = self.crop_to_roi(frame, bbox)
        # if roi.size == 0:
        #     logging.warning("Cropped ROI empty.")
        #     return "Empty ROI."

        url = self.convert_image_to_data_url(frame)
        raw = self.OCR_Inference(url) or ""
        candidate = re.sub(r'[^A-Z0-9]', '', raw.upper())

        if any(p.fullmatch(candidate) for p in PLATE_PATTERNS):
            self.save_ocr_result(candidate)
            return candidate
        else:
            logging.info(f"Rejected '{candidate}' (invalid format)")
            return f"Invalid: {candidate}"

    def batch_ocr(self, frames: list[np.ndarray]) -> list[str]:
        return [self.perform_ocr(f) for f in frames]


def process_image_for_ocr(image_path: str, ocr_instance: OCR3) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return ocr_instance.perform_ocr(img)