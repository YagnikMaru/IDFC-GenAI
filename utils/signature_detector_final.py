import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import re

class SignatureDetector:
    """Advanced signature and stamp detection for document processing."""

    def __init__(self, config):
        self.config = config

    def detect_signature(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect signature in document image using multiple methods."""
        if image is None or image.size == 0:
            return {"present": False, "bbox": [], "confidence": 0.0}

        # Method 1: Advanced rule-based detection
        result = self._detect_signature_advanced(image)

        if result["present"]:
            return result

        # Method 2: Text-based detection (if OCR results available)
        # For now, return the rule-based result
        return result

    def detect_stamp(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect stamp in document image using multiple methods."""
        if image is None or image.size == 0:
            return {"present": False, "bbox": [], "confidence": 0.0}

        # Method 1: Advanced rule-based detection
        result = self._detect_stamp_advanced(image)

        if result["present"]:
            return result

        # Method 2: Text-based detection
        return result

    def _detect_signature_advanced(self, image: np.ndarray) -> Dict[str, Any]:
        """Advanced rule-based signature detection."""
        height, width = image.shape[:2]

        # Convert to grayscale and apply preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)

        # Multi-scale edge detection
        edges_canny = cv2.Canny(enhanced, 30, 100)
        edges_sobel = cv2.Sobel(enhanced, cv2.CV_64F, 1, 1, ksize=3)
        edges_sobel = cv2.convertScaleAbs(edges_sobel)

        # Combine edges
        edges = cv2.bitwise_or(edges_canny, edges_sobel)

        # Morphological operations to connect signature strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        signature_candidates = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Size constraints for signatures (made more flexible)
            if not (15 < w < 400 and 8 < h < 120):
                continue

            # Aspect ratio (signatures are usually wider than tall, but allow some variation)
            aspect_ratio = w / h if h > 0 else 0
            if not (1.2 < aspect_ratio < 10.0):
                continue

            # Position constraints (usually in lower portion of document, but flexible)
            if y < height * 0.3:  # Must be in lower 70% of document
                continue

            # Area constraints
            area = cv2.contourArea(contour)
            if not (50 < area < 20000):
                continue

            # Extract ROI for detailed analysis
            roi = enhanced[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            # Calculate signature features
            features = self._calculate_signature_features(roi, contour)

            # Scoring based on multiple features
            score = self._calculate_signature_score(features)

            # Lower threshold for signature detection to catch more cases
            if score > 0.4:  # Reduced from 0.6
                signature_candidates.append({
                    "bbox": [int(x), int(y), int(x+w), int(y+h)],
                    "score": score,
                    "features": features,
                    "contour": contour
                })

        # Also try a simpler pattern-based approach
        simple_signature = self._detect_simple_signature_patterns(image)
        if simple_signature["present"]:
            signature_candidates.append({
                "bbox": simple_signature["bbox"],
                "score": simple_signature["confidence"] * 0.8,
                "features": {},
                "contour": None
            })

        if signature_candidates:
            # Select best candidate
            signature_candidates.sort(key=lambda x: x["score"], reverse=True)
            best = signature_candidates[0]

            # Convert score to confidence
            confidence = min(0.95, best["score"] * 1.2)

            return {
                "present": True,
                "bbox": best["bbox"],
                "confidence": confidence,
                "method": "advanced_rule_based",
                "score": best["score"]
            }

        return {"present": False, "bbox": [], "confidence": 0.0}

    def _detect_stamp_advanced(self, image: np.ndarray) -> Dict[str, Any]:
        """Advanced rule-based stamp detection."""
        height, width = image.shape[:2]

        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create color masks for common stamp colors
        # Red stamps
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)

        # Blue stamps (expanded range)
        lower_blue = np.array([85, 40, 40])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Green stamps (less common but possible)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Black/dark regions (embossed stamps)
        _, black_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        # Combine all color masks
        color_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, blue_mask), green_mask)
        combined_mask = cv2.bitwise_or(color_mask, black_mask)

        # Apply morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # Clean up noise
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        # Fill gaps
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_large)

        # Find contours
        contours, hierarchy = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        stamp_candidates = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Size constraints for stamps
            area = cv2.contourArea(contour)
            if not (300 < area < 30000):
                continue

            if not (25 < w < 200 and 25 < h < 200):
                continue

            # Aspect ratio (stamps are usually square or slightly rectangular)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 2.5:
                continue

            # Position constraints (can be anywhere but usually in corners or center)
            # Allow stamps anywhere in the document

            # Extract ROI for detailed analysis
            roi_color = image[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x:x+w]
            roi_mask = cleaned_mask[y:y+h, x:x+w]

            if roi_color.size == 0 or roi_gray.size == 0:
                continue

            # Calculate stamp features
            features = self._calculate_stamp_features(roi_color, roi_gray, roi_mask, contour)

            # Scoring based on multiple features
            score = self._calculate_stamp_score(features)

            if score > 0.5:  # Threshold for stamp detection
                stamp_candidates.append({
                    "bbox": [int(x), int(y), int(x+w), int(y+h)],
                    "score": score,
                    "features": features,
                    "contour": contour
                })

        if stamp_candidates:
            # Select best candidate
            stamp_candidates.sort(key=lambda x: x["score"], reverse=True)
            best = stamp_candidates[0]

            # Convert score to confidence
            confidence = min(0.95, best["score"] * 1.3)

            return {
                "present": True,
                "bbox": best["bbox"],
                "confidence": confidence,
                "method": "advanced_rule_based",
                "score": best["score"]
            }

        return {"present": False, "bbox": [], "confidence": 0.0}

    def _calculate_signature_features(self, roi: np.ndarray, contour) -> Dict[str, float]:
        """Calculate features for signature classification."""
        features = {}

        # Basic statistics
        features['mean_intensity'] = np.mean(roi)
        features['std_intensity'] = np.std(roi)
        features['variance'] = np.var(roi)

        # Edge density
        edges = cv2.Canny(roi, 50, 150)
        features['edge_density'] = np.sum(edges) / (255 * roi.size)

        # Contour complexity
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        features['complexity'] = perimeter / area if area > 0 else 0

        # Stroke width variation
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        features['stroke_density'] = np.sum(binary) / (255 * roi.size)

        # Local variance (signatures have varying stroke patterns)
        kernel_size = 5
        local_var = cv2.blur(roi.astype(np.float32), (kernel_size, kernel_size))
        features['local_variance'] = np.var(local_var)

        return features

    def _calculate_stamp_features(self, roi_color: np.ndarray, roi_gray: np.ndarray,
                                roi_mask: np.ndarray, contour) -> Dict[str, float]:
        """Calculate features for stamp classification."""
        features = {}

        # Color diversity
        if len(roi_color.shape) == 3:
            features['color_variance'] = np.var(roi_color.astype(np.float32))
            # Dominant color analysis
            hsv_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            features['hue_variance'] = np.var(hsv_roi[:, :, 0])
            features['saturation_mean'] = np.mean(hsv_roi[:, :, 1])
        else:
            features['color_variance'] = 0
            features['hue_variance'] = 0
            features['saturation_mean'] = 0

        # Edge definition
        edges = cv2.Canny(roi_gray, 50, 150)
        features['edge_density'] = np.sum(edges) / (255 * roi_gray.size)

        # Mask coverage
        features['mask_coverage'] = np.sum(roi_mask) / (255 * roi_mask.size)

        # Shape regularity
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        features['shape_regularity'] = area / rect_area if rect_area > 0 else 0

        # Texture analysis
        features['texture_contrast'] = np.std(roi_gray)

        return features

    def _calculate_signature_score(self, features: Dict[str, float]) -> float:
        """Calculate signature likelihood score."""
        score = 0

        # High variance indicates varied strokes (good for signatures)
        if features['variance'] > 500:
            score += 0.3
        elif features['variance'] > 200:
            score += 0.2

        # Moderate edge density
        if 0.05 < features['edge_density'] < 0.3:
            score += 0.25

        # Complex contour shape
        if features['complexity'] > 0.05:
            score += 0.2

        # Moderate stroke density
        if 0.2 < features['stroke_density'] < 0.8:
            score += 0.15

        # Local variance indicates varied patterns
        if features['local_variance'] > 100:
            score += 0.1

        return min(1.0, score)

    def _calculate_stamp_score(self, features: Dict[str, float]) -> float:
        """Calculate stamp likelihood score."""
        score = 0

        # Color variance (stamps often have multiple colors)
        if features['color_variance'] > 1000:
            score += 0.25

        # High saturation often indicates stamp inks
        if features['saturation_mean'] > 100:
            score += 0.2

        # Well-defined edges
        if features['edge_density'] > 0.1:
            score += 0.2

        # Good mask coverage
        if features['mask_coverage'] > 0.15:
            score += 0.15

        # Regular shape
        if features['shape_regularity'] > 0.7:
            score += 0.1

        # Texture contrast
        if features['texture_contrast'] > 30:
            score += 0.1

        return min(1.0, score)

    def _detect_simple_signature_patterns(self, image: np.ndarray) -> Dict[str, Any]:
        """Simple pattern-based signature detection for cursive writing."""
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Look for signature-like regions in the lower part of the document
        signature_region = gray[int(height * 0.6):, :]  # Lower 40% of document

        if signature_region.size == 0:
            return {"present": False, "bbox": [], "confidence": 0.0}

        # Apply adaptive thresholding to find text/stroke regions
        thresh = cv2.adaptiveThreshold(signature_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours in the signature region
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_signature = None
        best_score = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Adjust coordinates back to full image
            y += int(height * 0.6)

            # Size check for signature-like elements
            if not (30 < w < 250 and 10 < h < 80):
                continue

            aspect_ratio = w / h if h > 0 else 0
            if not (1.5 < aspect_ratio < 6.0):
                continue

            # Extract ROI
            roi = thresh[y - int(height * 0.6):y - int(height * 0.6) + h, x:x + w]

            # Calculate stroke features
            stroke_density = np.sum(roi) / (255 * roi.size)

            # Look for connected components (signature has multiple strokes)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi)

            # Signatures typically have multiple connected components
            if num_labels < 3:
                continue

            # Calculate component size variation (signatures have varied stroke lengths)
            if len(stats) > 1:
                component_sizes = stats[1:, cv2.CC_STAT_AREA]  # Skip background
                size_std = np.std(component_sizes) if len(component_sizes) > 1 else 0
                size_mean = np.mean(component_sizes)

                # Score based on stroke variation and density
                score = (
                    min(0.4, stroke_density * 2) +  # Moderate stroke density
                    min(0.4, size_std / 500) +      # Stroke size variation
                    min(0.2, (num_labels - 2) / 10) # Number of strokes
                )

                if score > best_score and score > 0.3:
                    best_score = score
                    best_signature = {
                        "bbox": [x, y, x + w, y + h],
                        "score": score
                    }

        if best_signature:
            return {
                "present": True,
                "bbox": best_signature["bbox"],
                "confidence": min(0.9, best_signature["score"] * 1.5)
            }

        return {"present": False, "bbox": [], "confidence": 0.0}