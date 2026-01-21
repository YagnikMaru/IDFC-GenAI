import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Config:
    # File paths
    TRAIN_DIR = "train"
    OUTPUT_DIR = "sample_output"
    MODEL_DIR = "models"
    
    # OCR settings
    OCR_ENGINE = "paddleocr"  # Options: paddleocr, easyocr, tesseract
    LANGUAGES = ['en', 'hi', 'gu']  # English, Hindi, Gujarati
    OCR_THRESHOLD = 0.7
    
    # Field extraction settings
    DEALER_NAME_KEYWORDS = ["dealer", "seller", "company", "corporation", "ltd", "pvt", "limited"]
    MODEL_KEYWORDS = ["model", "tractor", "description", "product"]
    HORSEPOWER_KEYWORDS = ["horse power", "hp", "power", "engine"]
    COST_KEYWORDS = ["total", "amount", "cost", "price", "grand total", "rs"]
    
    # Signature detection
    SIGNATURE_DETECTOR_MODEL = os.path.join(MODEL_DIR, "signature_detector.pth")
    STAMP_DETECTOR_MODEL = os.path.join(MODEL_DIR, "stamp_detector.pth")
    DETECTION_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    
    # Performance
    MAX_PROCESSING_TIME = 30  # seconds
    TARGET_ACCURACY = 0.95
    
    # Output format
    OUTPUT_JSON_SCHEMA = {
        "doc_id": str,
        "fields": {
            "dealer_name": str,
            "model_name": str,
            "horse_power": int,
            "asset_cost": int,
            "signature": {"present": bool, "bbox": List[int]},
            "stamp": {"present": bool, "bbox": List[int]}
        },
        "confidence": float,
        "processing_time_sec": float,
        "cost_estimate_usd": float
    }