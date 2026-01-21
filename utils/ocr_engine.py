import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
import re
from PIL import Image
import pdf2image
import warnings
warnings.filterwarnings('ignore')

class OCREngine:
    """OCR engine supporting multiple languages and document types."""
    
    def __init__(self, config):
        self.config = config
        self.engine = None
        self._initialize_engine()
        
    def _initialize_engine(self):
        """Initialize the selected OCR engine with error handling."""
        engine_type = self.config.OCR_ENGINE
        
        try:
            if engine_type == "paddleocr":
                try:
                    from paddleocr import PaddleOCR
                    # Try newer API first
                    self.engine = PaddleOCR(
                        lang='en',
                        use_angle_cls=True,
                        show_log=False,
                        use_gpu=False
                    )
                except TypeError as e:
                    # If new API fails, try older API
                    print(f"Warning: PaddleOCR API issue: {e}")
                    print("Trying alternative initialization...")
                    try:
                        self.engine = PaddleOCR(
                            lang='en',
                            use_angle_cls=True,
                            show_log=False,
                            gpu_mem=0  # Force CPU
                        )
                    except Exception as e2:
                        print(f"PaddleOCR failed: {e2}")
                        raise
                        
            elif engine_type == "easyocr":
                try:
                    import easyocr
                    self.engine = easyocr.Reader(
                        self.config.LANGUAGES,
                        gpu=False
                    )
                except Exception as e:
                    print(f"EasyOCR failed: {e}")
                    raise
                    
            elif engine_type == "tesseract":
                try:
                    import pytesseract
                    self.engine = pytesseract
                    # Check if tesseract is installed
                    import subprocess
                    subprocess.run(['tesseract', '--version'], 
                                 capture_output=True, check=True)
                except Exception as e:
                    print(f"Tesseract failed: {e}")
                    raise
            else:
                raise ValueError(f"Unsupported OCR engine: {engine_type}")
                
        except Exception as e:
            print(f"Error initializing OCR engine {engine_type}: {e}")
            print("Falling back to available engine...")
            self._try_fallback_engines()
    
    def _try_fallback_engines(self):
        """Try to initialize fallback OCR engines."""
        engines_to_try = ['easyocr', 'tesseract', 'paddleocr']
        
        for engine_name in engines_to_try:
            try:
                if engine_name == "easyocr":
                    import easyocr
                    self.engine = easyocr.Reader(['en'], gpu=False)
                    self.config.OCR_ENGINE = "easyocr"
                    print(f"Successfully initialized {engine_name}")
                    break
                elif engine_name == "tesseract":
                    import pytesseract
                    self.engine = pytesseract
                    self.config.OCR_ENGINE = "tesseract"
                    print(f"Successfully initialized {engine_name}")
                    break
            except Exception as e:
                print(f"{engine_name} failed: {e}")
                continue
        
        if self.engine is None:
            print("Warning: No OCR engine could be initialized.")
            print("The system will run with limited functionality.")
    
    def extract_text_and_layout(self, image) -> Dict[str, Any]:
        """
        Extract text and layout information from image.
        
        Returns:
            Dictionary containing text, bounding boxes, and confidence scores
        """
        if self.engine is None:
            print("No OCR engine available.")
            return {"text": [], "boxes": [], "confidences": [], "avg_confidence": 0.0}
        
        try:
            # Try primary engine first
            result = self._extract_with_engine(image, self.config.OCR_ENGINE)
            
            # If confidence is too low, try fallback engines
            if result.get("avg_confidence", 0.0) < 0.6 and len(result.get("text", [])) < 5:
                print(f"Low confidence with {self.config.OCR_ENGINE}, trying fallbacks...")
                
                fallback_engines = [e for e in ['paddleocr', 'easyocr', 'tesseract'] 
                                  if e != self.config.OCR_ENGINE]
                
                for fallback_engine in fallback_engines:
                    try:
                        fallback_result = self._extract_with_engine(image, fallback_engine)
                        if (fallback_result.get("avg_confidence", 0.0) > result.get("avg_confidence", 0.0) 
                            or len(fallback_result.get("text", [])) > len(result.get("text", []))):
                            print(f"Using {fallback_engine} as it performed better")
                            result = fallback_result
                            break
                    except Exception as e:
                        print(f"Fallback engine {fallback_engine} failed: {e}")
                        continue
            
            return result
            
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return {"text": [], "boxes": [], "confidences": [], "avg_confidence": 0.0}
    
    def _extract_with_engine(self, image, engine_name: str) -> Dict[str, Any]:
        """Extract text using a specific OCR engine."""
        # Temporarily change config
        original_engine = self.config.OCR_ENGINE
        self.config.OCR_ENGINE = engine_name
        
        try:
            if engine_name == "paddleocr":
                return self._extract_with_paddleocr(image)
            elif engine_name == "easyocr":
                return self._extract_with_easyocr(image)
            elif engine_name == "tesseract":
                return self._extract_with_tesseract(image)
            else:
                return {"text": [], "boxes": [], "confidences": [], "avg_confidence": 0.0}
        finally:
            # Restore original config
            self.config.OCR_ENGINE = original_engine
    
    def _extract_with_paddleocr(self, image):
        """Extract text using PaddleOCR."""
        try:
            # Convert image to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            result = self.engine.ocr(image, cls=True)
            
            if not result or not result[0]:
                return {"text": [], "boxes": [], "confidences": [], "avg_confidence": 0.0}
            
            texts, boxes, confidences = [], [], []
            
            for line in result[0]:
                if line and len(line) >= 2:
                    box = line[0]
                    text_info = line[1]
                    
                    if len(text_info) >= 2:
                        text = text_info[0]
                        confidence = float(text_info[1])
                        
                        if confidence >= self.config.OCR_THRESHOLD:
                            texts.append(text)
                            # Flatten bounding box coordinates
                            flat_box = []
                            for point in box:
                                flat_box.extend([int(point[0]), int(point[1])])
                            boxes.append(flat_box)
                            confidences.append(confidence)
            
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                "text": texts,
                "boxes": boxes,
                "confidences": confidences,
                "avg_confidence": avg_confidence,
                "language": "detected",
                "engine": "paddleocr"
            }
        except Exception as e:
            print(f"PaddleOCR extraction error: {e}")
            return {"text": [], "boxes": [], "confidences": [], "avg_confidence": 0.0}
    
    def _extract_with_easyocr(self, image):
        """Extract text using EasyOCR."""
        try:
            results = self.engine.readtext(image)
            
            texts, boxes, confidences = [], [], []
            
            for result in results:
                if len(result) >= 3:
                    box = result[0]
                    text = result[1]
                    confidence = result[2]
                    
                    if confidence >= self.config.OCR_THRESHOLD:
                        texts.append(text)
                        boxes.append([int(coord) for point in box for coord in point])
                        confidences.append(confidence)
            
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                "text": texts,
                "boxes": boxes,
                "confidences": confidences,
                "avg_confidence": avg_confidence,
                "language": "detected",
                "engine": "easyocr"
            }
        except Exception as e:
            print(f"EasyOCR extraction error: {e}")
            return {"text": [], "boxes": [], "confidences": [], "avg_confidence": 0.0}
    
    def _extract_with_tesseract(self, image):
        """Extract text using Tesseract."""
        try:
            # Convert image to PIL format
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    image_pil = Image.fromarray(image)
                else:
                    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = Image.open(image)
            
            # Get OCR data
            data = self.engine.image_to_data(
                image_pil,
                output_type=self.engine.Output.DICT,
                lang='eng'  # Start with English only
            )
            
            texts, boxes, confidences = [], [], []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0 and data['text'][i].strip():
                    confidence = float(data['conf'][i]) / 100.0
                    
                    if confidence >= self.config.OCR_THRESHOLD:
                        texts.append(data['text'][i])
                        boxes.append([
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ])
                        confidences.append(confidence)
            
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                "text": texts,
                "boxes": boxes,
                "confidences": confidences,
                "avg_confidence": avg_confidence,
                "language": "eng",
                "engine": "tesseract"
            }
        except Exception as e:
            print(f"Tesseract extraction error: {e}")
            return {"text": [], "boxes": [], "confidences": [], "avg_confidence": 0.0}
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR results."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply bilateral filter to reduce noise while keeping edges sharp
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(filtered)
            
            # Apply different thresholding based on OCR engine
            if self.config.OCR_ENGINE in ['tesseract', 'easyocr']:
                # Tesseract works better with inverted black text on white background
                _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh = 255 - thresh  # Invert for black text on white background
            else:
                # PaddleOCR prefers original contrast
                thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
            
            # Morphological operations to clean up the image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Resize if too small or too large
            height, width = cleaned.shape
            if max(height, width) < 500:
                scale = 500 / max(height, width)
                cleaned = cv2.resize(cleaned, None, fx=scale, fy=scale, 
                                   interpolation=cv2.INTER_CUBIC)
            elif max(height, width) > 4000:
                scale = 4000 / max(height, width)
                cleaned = cv2.resize(cleaned, None, fx=scale, fy=scale, 
                                   interpolation=cv2.INTER_AREA)
            
            # Convert back to 3 channels if needed
            if self.config.OCR_ENGINE in ['paddleocr', 'easyocr']:
                cleaned = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            
            return cleaned
            
        except Exception as e:
            print(f"Image preprocessing error: {e}")
            return image
    
    def detect_language(self, image):
        """Detect language in the document."""
        try:
            # Simple language detection based on common patterns
            result = self.extract_text_and_layout(image)
            text = ' '.join(result.get('text', []))
            
            if not text:
                return 'en'
            
            # Count Devanagari characters (Hindi)
            devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
            # Count Gujarati characters
            gujarati_count = sum(1 for char in text if '\u0A80' <= char <= '\u0AFF')
            
            total_chars = len([c for c in text if c.isalpha()])
            
            if total_chars == 0:
                return 'en'
            
            if gujarati_count / total_chars > 0.3:
                return 'gu'
            elif devanagari_count / total_chars > 0.3:
                return 'hi'
            else:
                return 'en'
                
        except Exception as e:
            print(f"Language detection error: {e}")
            return 'en'