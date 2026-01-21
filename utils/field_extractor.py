import re
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class FieldExtractor:
    """Extract specific fields from OCR results using rules and heuristics."""
    
    def __init__(self, config):
        self.config = config
        self.regex_patterns = self._initialize_regex_patterns()
        
    def _initialize_regex_patterns(self):
        """Initialize regex patterns for field extraction."""
        return {
            "dealer_name": [
                r"(?i)(?:[A-Z][A-Z\s&,.]*(?:LIMITED|LTD|PVT|PRIVATE|CORPORATION|CORP|COMPANY|CO\.?|INC\.?))",
                r"(?i)^[A-Z][A-Z\s&,.]{10,}(?:LIMITED|LTD|PVT|PRIVATE|CORPORATION|CORP|COMPANY|CO\.?|INC\.?)?",
                r"(?i)(?:M\/s\.?|S\/o|D\/o|Smt\.?|Shri\.?|Mr\.?|Ms\.?)[\s:]*([A-Za-z\s.,&]+(?:LIMITED|LTD|PVT|PRIVATE|CORPORATION|CORP|COMPANY|CO\.?|INC\.?)?)",
                r"(?i)(?:DEALER|SELLER|VENDOR|SOLD BY)[\s:]*([A-Z][A-Z\s&,.]+)",
                r"(?i)(?:FROM|BY)[\s:]*([A-Z][A-Z\s&,.]+(?:LIMITED|LTD|PVT|PRIVATE|CORPORATION|CORP|COMPANY|CO\.?|INC\.?)?)"
            ],
            "model_name": [
                r"(?i)(?:tractor|model|description|product|item)[\s:]*([A-Z0-9][A-Z0-9\s\-/]+[A-Z0-9](?:4WD|2WD|DI|XPTO?)?)",
                r"\b[A-Z][A-Z0-9\s\-/]{3,}(?:4WD|2WD|DI|XPTO?|PLUS|PRO|PREMIUM)\b",
                r"\b(?:KUBOTA|MAHINDRA|SWARAJ|ESCORTS|JOHN DEERE|NEW HOLLAND|FARMTRAC|SONALIKA|TATA|CASE IH|CLAAS|FIAT|SAME|DEUTZ-FAHR)[\s\-]*[A-Z0-9]+\b",
                r"(?i)(?:MODEL|TYPE|VARIANT)[\s:]*([A-Z0-9][A-Z0-9\s\-/]+)",
                r"\b[A-Z]{2,}[\-_]?[0-9]{2,}[\-_]?[A-Z0-9]*\b"  # Pattern like MAHINDRA-275-DI
            ],
            "horse_power": [
                r"(\d{2,3})\s*(?:HP|H\.?P\.?|HORSE\s*POWER|KW|KILOWATT)",
                r"(?:POWER|ENGINE|CAPACITY)[\s:]*(\d{2,3})\s*(?:HP|KW)?",
                r"\b(\d{2,3})\s*HP\b",
                r"(?:HORSEPOWER|HP RATING)[\s:]*[:\-]*\s*(\d{2,3})",
                r"(\d{2,3})\s*(?:BHP|PS|CV)",  # Brake Horse Power, Pferdestärke, Cheval Vapeur
                r"(?:TRACTOR|MODEL)[\s:]*(\d{2,3})\s*HP",  # Tractor X HP
                r"(\d{2,3})\s*HP\s*(?:TRACTOR|ENGINE)",  # X HP Tractor
                r"(\d{1,2})[\s\-](\d{1,2})\s*HP",  # X-Y HP range, take first number
            ],
            "asset_cost": [
                r"(?:TOTAL|AMOUNT|COST|PRICE|GRAND TOTAL|NET TOTAL|INVOICE TOTAL)[\s:]*[RS\.Rs₹]*\s*([\d,]+(?:\.\d{2})?)",
                r"RS\.\s*([\d,]+(?:\.\d{2})?)(?:\s*/-|\s*ONLY)?",
                r"[₹RS]\s*([\d,]+(?:\.\d{2})?)",
                r"\b([\d,]+(?:\.\d{2})?)\s*(?:/-|RS|₹|RUPEES)",
                r"(?:AMOUNT|COST|PRICE)[\s:]*[:\-]*\s*[RS\.Rs₹]*\s*([\d,]+(?:\.\d{2})?)",
                r"(?:TOTAL VALUE|ASSET VALUE)[\s:]*([\d,]+(?:\.\d{2})?)"
            ],
            "currency": r"(?:RS|₹|INR|USD|EUR|GBP|RUPEES)"
        }
    
    def extract_fields(self, ocr_result: Dict) -> Dict[str, Any]:
        """Extract all required fields from OCR results."""
        fields = {}
        
        # Combine all text with their positions
        text_elements = list(zip(
            ocr_result.get("text", []),
            ocr_result.get("boxes", []),
            ocr_result.get("confidences", [])
        ))
        
        # Extract dealer name
        fields["dealer_name"] = self._extract_dealer_name(text_elements)
        
        # Extract model name
        fields["model_name"] = self._extract_model_name(text_elements)
        
        # Extract horse power
        fields["horse_power"] = self._extract_horse_power(text_elements)
        
        # Extract asset cost
        fields["asset_cost"] = self._extract_asset_cost(text_elements)
        
        return fields
    
    def _extract_dealer_name(self, text_elements: List[Tuple]) -> str:
        """Extract dealer/company name."""
        candidates = []
        
        for text, bbox, confidence in text_elements:
            text = str(text).strip()
            
            # Skip very short texts or numbers
            if len(text) < 3 or text.isdigit():
                continue
            
            # Look for company indicators with higher priority
            company_indicators = ["LTD", "PVT", "CORP", "COMPANY", "LIMITED", "PRIVATE", "CORPORATION", "INC"]
            has_company_indicator = any(indicator in text.upper() for indicator in company_indicators)
            
            # Check regex patterns
            for pattern in self.regex_patterns["dealer_name"]:
                match = re.search(pattern, text)
                if match:
                    # Extract the actual company name from the match
                    extracted_name = match.group(1) if match.groups() else match.group(0)
                    extracted_name = extracted_name.strip()
                    
                    # Calculate score based on various factors
                    score = confidence
                    
                    # Boost score for company indicators
                    if has_company_indicator:
                        score += 0.3
                    
                    # Boost score for proper capitalization
                    if extracted_name.istitle() or extracted_name.isupper():
                        score += 0.1
                    
                    # Boost score for reasonable length
                    if 5 <= len(extracted_name) <= 100:
                        score += 0.1
                    
                    # Penalize for very long texts (likely full paragraphs)
                    if len(extracted_name) > 150:
                        score -= 0.2
                    
                    candidates.append((extracted_name, score, bbox[1]))
            
            # Also check for company-like patterns without regex match
            # Look for text that ends with company indicators
            if has_company_indicator:
                candidates.append((text, confidence * 0.9, bbox[1]))
            
            # Look for text that starts with "The" and contains company words
            if text.upper().startswith("THE ") and any(word in text.upper() for word in ["CORPORATION", "COMPANY", "LIMITED", "LTD"]):
                candidates.append((text, confidence * 0.8, bbox[1]))
            
            # Look for any text containing corporation/company/limited
            if any(word in text.upper() for word in ["CORPORATION", "COMPANY", "LIMITED", "LTD", "PRIVATE", "INDUSTRIES"]):
                candidates.append((text, confidence * 0.7, bbox[1]))
        
        if candidates:
            # Sort by score (descending), then by position (top first)
            candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            
            # Return the top candidate, but check if it's reasonable
            top_candidate = candidates[0][0]
            if self._is_valid_dealer_name(top_candidate):
                return top_candidate
        
        # Fallback: find text at top of document (headers often contain dealer info)
        top_texts = []
        for text, bbox, conf in text_elements:
            text = str(text).strip()
            if bbox[1] < 300 and len(text) > 3:  # Top 300 pixels
                # Look for company-like text in header
                if any(indicator in text.upper() for indicator in ["LTD", "CORP", "COMPANY", "LIMITED", "CORPORATION", "INDUSTRIES"]):
                    top_texts.append((text, conf, bbox[1]))
                elif text.upper().startswith("THE ") and len(text) > 10:
                    top_texts.append((text, conf * 0.7, bbox[1]))
                elif len(text) > 15 and not text.isdigit() and not text.replace('.', '').replace(' ', '').isdigit():
                    # Any reasonably long text in header could be company name
                    top_texts.append((text, conf * 0.5, bbox[1]))
        
        if top_texts:
            top_texts.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            return top_texts[0][0]
        
        return ""
    
    def _is_valid_dealer_name(self, name: str) -> bool:
        """Check if a dealer name candidate is valid."""
        if not name or len(name) < 3:
            return False
        
        # Check for too many numbers (likely not a company name)
        number_count = sum(c.isdigit() for c in name)
        if number_count > len(name) * 0.3:  # More than 30% numbers
            return False
        
        # Check for common invalid patterns
        invalid_patterns = [
            r'^\d+',  # Starts with number
            r'^[a-z\s]+$',  # All lowercase (likely not a proper name)
            r'^\W+',  # Starts with special character
            r'(?:total|amount|price|date|invoice|bill)$',  # Common field names
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return False
        
        return True
    
    def _extract_model_name(self, text_elements: List[Tuple]) -> str:
        """Extract tractor/model name."""
        candidates = []
        
        for text, bbox, confidence in text_elements:
            text = str(text).strip().upper()
            
            # Skip very short texts or pure numbers
            if len(text) < 2 or text.isdigit():
                continue
            
            # Look for tractor-related keywords with context
            tractor_keywords = ["TRACTOR", "MODEL", "DESCRIPTION", "PRODUCT", "ITEM", "TYPE", "VARIANT"]
            has_tractor_context = any(kw in text for kw in tractor_keywords)
            
            # Check regex patterns
            for pattern in self.regex_patterns["model_name"]:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    extracted_model = match.group(1) if match.groups() else match.group(0)
                    extracted_model = extracted_model.strip().upper()
                    
                    # Calculate score
                    score = confidence
                    
                    # Boost score for tractor context
                    if has_tractor_context:
                        score += 0.3
                    
                    # Boost score for known tractor brands
                    known_brands = ["KUBOTA", "MAHINDRA", "SWARAJ", "ESCORTS", "JOHN DEERE", 
                                  "NEW HOLLAND", "FARMTRAC", "SONALIKA", "TATA", "CASE IH"]
                    if any(brand in extracted_model for brand in known_brands):
                        score += 0.2
                    
                    # Boost score for model-like patterns (alphanumeric with dashes)
                    if re.search(r'[A-Z0-9]+[\-_][0-9]+', extracted_model):
                        score += 0.1
                    
                    # Reasonable length check
                    if 3 <= len(extracted_model) <= 50:
                        score += 0.1
                    
                    candidates.append((extracted_model, score, bbox[1]))
            
            # Also consider standalone model numbers/patterns
            if re.match(r'^[A-Z0-9\-/]{3,20}$', text) and not text.isdigit():
                # Check if it looks like a model number
                if re.search(r'[A-Z]+[0-9]|[0-9]+[A-Z]', text):
                    candidates.append((text, confidence * 0.7, bbox[1]))
            
            # Look for any text that might be a model (contains numbers and letters)
            if re.search(r'[A-Z]{2,}.*[0-9]|[0-9].*[A-Z]{2,}', text.upper()) and len(text) > 3 and len(text) < 30:
                # Avoid dates and pure numbers
                if not re.match(r'^\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4}$', text):  # Not a date
                    if not re.match(r'^\d+\.\d+$', text):  # Not just a decimal number like price
                        candidates.append((text, confidence * 0.6, bbox[1]))
            
            # Special case: alphanumeric codes that might be model numbers
            if re.match(r'^\d+\.\d{5}\.\d{2}$', text):  # Pattern like 8.01815.00
                candidates.append((text, confidence * 0.8, bbox[1]))
        
        if candidates:
            # Sort by score (descending), then by position
            candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            
            # Return top candidate if it passes validation
            top_candidate = candidates[0][0]
            if self._is_valid_model_name(top_candidate):
                return top_candidate
        
        return ""
    
    def _is_valid_model_name(self, model: str) -> bool:
        """Check if a model name candidate is valid."""
        if not model or len(model) < 2:
            return False
        
        # Should have at least one letter OR be a valid numeric code
        has_letters = any(c.isalpha() for c in model)
        is_numeric_code = bool(re.match(r'^\d+\.\d+\.\d+$', model))  # Like 8.01815.00
        
        if not (has_letters or is_numeric_code):
            return False
        
        # Should not be all special characters
        if not re.search(r'[A-Z0-9]', model):
            return False
        
        # Check for invalid patterns
        invalid_patterns = [
            r'^(?:TOTAL|AMOUNT|PRICE|DATE|INVOICE|BILL|DEALER|SELLER)$',  # Common field names
            r'^\d{1,2}$',  # Single or double digits (likely HP or other fields)
            r'^[A-Z]{1,2}$',  # Very short all caps (likely abbreviations)
            r'^\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4}$',  # Date pattern DD.MM.YYYY
            r'^\d{4}[\./\-]\d{1,2}[\./\-]\d{1,2}$',  # Date pattern YYYY.MM.DD
            r'.*\bDATE\b.*',  # Contains "DATE"
            r'.*\bAMOUNT\b.*',  # Contains "AMOUNT"
            r'.*\bTOTAL\b.*',   # Contains "TOTAL"
            r'.*\bRUPEES?\b.*', # Contains "RUPEE"
            r'.*\bOFFICE\b.*',  # Contains "OFFICE"
            r'.*\bBANK\b.*',    # Contains "BANK"
            r'.*\bFINANCED?\b.*', # Contains "FINANCE"
            r'.*\bPHONE?\b.*',  # Contains "PHONE"
            r'.*\bPH:?\b.*',    # Contains "PH:"
            r'.*\bMOBILE?\b.*', # Contains "MOBILE"
            r'.*\bCONTACT?\b.*', # Contains "CONTACT"
            r'^\s*LTD\s*$',     # Just "LTD"
            r'^\s*PVT\s*$',     # Just "PVT"
            r'^\s*CO\s*$',      # Just "CO"
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, model):
                return False
        
        return True
    
    def _extract_horse_power(self, text_elements: List[Tuple]) -> int:
        """Extract horse power value."""
        candidates = []
        
        for text, bbox, confidence in text_elements:
            text = str(text).strip().upper()
            
            # Check regex patterns
            for pattern in self.regex_patterns["horse_power"]:
                matches = re.findall(pattern, text)
                for match in matches:
                    try:
                        # Extract numeric part
                        if isinstance(match, tuple):
                            hp_value = int(re.sub(r'\D', '', match[0]))
                        else:
                            hp_value = int(re.sub(r'\D', '', match))
                        
                        # Validate range (reasonable tractor HP: 10-200)
                        if 10 <= hp_value <= 200:
                            # Calculate confidence score
                            score = confidence
                            
                            # Boost score if "HP" is explicitly mentioned
                            if "HP" in text:
                                score += 0.2
                            
                            # Boost score for power-related context
                            power_keywords = ["HORSEPOWER", "POWER", "ENGINE", "CAPACITY"]
                            if any(kw in text for kw in power_keywords):
                                score += 0.1
                            
                            candidates.append((hp_value, score, bbox[1]))
                            
                    except (ValueError, IndexError):
                        continue
        
        if candidates:
            # Sort by score (descending), then by position
            candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            
            # Return the highest scoring valid HP value
            return candidates[0][0]
        
        # Fallback: look for standalone numbers in power-related contexts or reasonable HP ranges
        for text, bbox, confidence in text_elements:
            text = str(text).strip().upper()
            
            # Look for "HP" followed by number
            hp_match = re.search(r'HP[\s:]*(\d{2,3})', text)
            if hp_match:
                try:
                    hp_value = int(hp_match.group(1))
                    if 10 <= hp_value <= 200:
                        return hp_value
                except:
                    continue
            
            # Look for number followed by "HP"
            hp_match = re.search(r'(\d{2,3})[\s:]*HP', text)
            if hp_match:
                try:
                    hp_value = int(hp_match.group(1))
                    if 10 <= hp_value <= 200:
                        return hp_value
                except:
                    continue
            
            # Look for standalone numbers that could be HP (in reasonable range)
            if text.isdigit():
                try:
                    num = int(text)
                    if 10 <= num <= 200:
                        # Check if it's near power-related keywords in other text elements
                        for other_text, other_bbox, _ in text_elements:
                            if (abs(bbox[1] - other_bbox[1]) < 50 and  # Same row approximately
                                any(kw in str(other_text).upper() for kw in ["POWER", "HORSE", "ENGINE", "HP"])):
                                return num
                except:
                    continue
        
        return 0
    
    def _extract_asset_cost(self, text_elements: List[Tuple]) -> int:
        """Extract asset cost/numeric value."""
        candidates = []
        
        for text, bbox, confidence in text_elements:
            text = str(text).strip()
            
            # Look for currency indicators and cost-related keywords
            currency_indicators = ["RS", "₹", "INR", "RUPEES"]
            cost_keywords = ["TOTAL", "AMOUNT", "COST", "PRICE", "GRAND TOTAL", "NET TOTAL", 
                           "INVOICE TOTAL", "ASSET VALUE", "TOTAL VALUE"]
            
            has_currency = any(indicator in text.upper() for indicator in currency_indicators)
            has_cost_keyword = any(kw in text.upper() for kw in cost_keywords)
            
            # Check regex patterns
            for pattern in self.regex_patterns["asset_cost"]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        # Clean the numeric string
                        num_str = str(match).replace(',', '').replace(' ', '')
                        
                        # Handle decimal points
                        if '.' in num_str:
                            # Check if it's rupees with paise (like 1,23,456.78)
                            parts = num_str.split('.')
                            if len(parts) == 2 and len(parts[1]) <= 2:
                                num = float(num_str)
                            else:
                                # Likely not a currency amount, take integer part
                                num = int(parts[0])
                        else:
                            num = int(num_str)
                        
                        # Validate reasonable tractor cost range (10,000 to 10,00,00,000)
                        # But exclude phone numbers and other invalid patterns
                        num_str_clean = re.sub(r'[^\d]', '', str(num))
                        
                        # Skip if it looks like a phone number (10+ digits, or contains phone patterns)
                        if len(num_str_clean) >= 10:  # Likely a phone number or account number
                            continue
                        
                        # Skip if the original text contains phone number patterns
                        if re.search(r'\(\d{4,5}\)\d{5,6}', text):  # (XXXXX)XXXXXX pattern
                            continue
                        if re.search(r'\d{4,5}-\d{5,6}', text):  # XXXX-XXXXXX pattern
                            continue
                        if 'phone' in text.lower() or 'ph:' in text.lower():
                            continue
                        
                        # Also check nearby text elements for phone context
                        phone_context = False
                        for other_text, other_bbox, _ in text_elements:
                            other_text = str(other_text).strip()
                            if (abs(bbox[1] - other_bbox[1]) < 30 and  # Close vertically
                                ('phone' in other_text.lower() or 'ph:' in other_text.lower() or 
                                 re.search(r'\(\d{4,5}\)', other_text))):  # Nearby phone indicator
                                phone_context = True
                                break
                        if phone_context:
                            continue
                            
                        if 10000 <= num <= 100000000:
                            # Calculate confidence score
                            score = confidence
                            
                            # Boost score for currency indicators
                            if has_currency:
                                score += 0.3
                            
                            # Boost score for cost keywords
                            if has_cost_keyword:
                                score += 0.2
                            
                            # Boost score for larger amounts (likely totals)
                            if num >= 100000:  # 1 lakh and above
                                score += 0.1
                            
                            # Penalize very small amounts that might be line items
                            if num < 50000:
                                score -= 0.1
                            
                            candidates.append((num, score, bbox[1]))
                            
                    except (ValueError, IndexError):
                        continue
            
            # Also check for standalone large numbers with currency context
            if has_currency:
                # Extract all numbers from the text
                numbers = re.findall(r'[\d,]+(?:\.\d{2})?', text)
                for num_str in numbers:
                    try:
                        num = int(num_str.replace(',', ''))
                        if 10000 <= num <= 100000000:
                            candidates.append((num, confidence * 0.8, bbox[1]))
                    except:
                        continue
        
        if candidates:
            # Sort by score (descending), then by amount (prefer larger amounts for totals)
            candidates.sort(key=lambda x: (x[1], x[0], -x[2]), reverse=True)
            
            # Return the highest scoring cost value
            return candidates[0][0]
        
        # Fallback: look for the largest reasonable number in the document
        all_numbers = []
        for text, bbox, confidence in text_elements:
            text = str(text).strip()
            numbers = re.findall(r'[\d,]+', text)
            for num_str in numbers:
                try:
                    num = int(num_str.replace(',', ''))
                    if 10000 <= num <= 100000000:
                        all_numbers.append((num, confidence, bbox[1]))
                except:
                    continue
        
        if all_numbers:
            # Take the largest number (usually the total)
            all_numbers.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return all_numbers[0][0]
        
        return 0
    
    def find_key_value_pairs(self, text_elements: List[Tuple]) -> Dict[str, str]:
        """Extract key-value pairs from document."""
        key_value_pairs = {}
        
        # Sort by y-coordinate (top to bottom), then x-coordinate (left to right)
        sorted_elements = sorted(text_elements, key=lambda x: (x[1][1], x[1][0]))
        
        for i, (text1, bbox1, _) in enumerate(sorted_elements):
            text1_lower = str(text1).lower()
            
            # Check if this looks like a key
            if any(keyword in text1_lower for keyword in self.config.DEALER_NAME_KEYWORDS + 
                   self.config.MODEL_KEYWORDS + self.config.HORSEPOWER_KEYWORDS + 
                   self.config.COST_KEYWORDS):
                
                # Look for value in nearby elements
                for j, (text2, bbox2, _) in enumerate(sorted_elements[i+1:i+5]):
                    # Check if text2 is to the right and at similar y-level
                    if (bbox2[0] > bbox1[2] and 
                        abs(bbox2[1] - bbox1[1]) < 50):
                        key_value_pairs[text1] = text2
                        break
        
        return key_value_pairs