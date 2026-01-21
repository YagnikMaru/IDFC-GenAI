import re
from typing import Dict, List, Any
from fuzzywuzzy import fuzz
import numpy as np

class PostProcessor:
    """Post-process extracted fields for consistency and validation."""
    
    def __init__(self, config):
        self.config = config
        
    def validate_and_process(self, 
                           text_fields: Dict, 
                           signature_result: Dict,
                           stamp_result: Dict) -> Dict[str, Any]:
        """Validate and post-process all extracted fields."""
        processed = {}
        
        # Process dealer name with fuzzy matching
        processed["dealer_name"] = self._clean_and_validate_dealer_name(text_fields.get("dealer_name", ""))
        
        # Process model name
        processed["model_name"] = self._clean_and_validate_model_name(text_fields.get("model_name", ""))
        
        # Process horse power with range validation
        processed["horse_power"] = self._validate_horse_power(text_fields.get("horse_power", 0))
        
        # Process asset cost with currency validation
        processed["asset_cost"] = self._validate_asset_cost(text_fields.get("asset_cost", 0))
        
        # Add signature and stamp with confidence validation
        processed["signature"] = self._validate_signature(signature_result)
        processed["stamp"] = self._validate_stamp(stamp_result)
        
        return processed
    
    def _clean_and_validate_dealer_name(self, dealer_name: str) -> str:
        """Clean and validate dealer name with fuzzy matching."""
        if not dealer_name:
            return ""
        
        # Clean the name first
        cleaned_name = self._clean_dealer_name(dealer_name)
        
        # Validate basic criteria
        if not self._is_valid_dealer_name(cleaned_name):
            return ""
        
        # Try fuzzy matching with master list if available
        # For now, we'll implement a basic validation
        # In production, you'd load a master dealer list
        
        return cleaned_name
    
    def _clean_dealer_name(self, dealer_name: str) -> str:
        """Clean and standardize dealer name."""
        if not dealer_name:
            return ""
        
        # Convert to uppercase for consistency
        dealer_name = dealer_name.upper()
        
        # Remove common prefixes
        prefixes = ["DEALER:", "FROM:", "SOLD BY:", "VENDOR:", "BY:"]
        for prefix in prefixes:
            if dealer_name.startswith(prefix):
                dealer_name = dealer_name[len(prefix):].strip()
        
        # Remove extra whitespace and special characters (but keep company indicators)
        dealer_name = re.sub(r'\s+', ' ', dealer_name)
        dealer_name = re.sub(r'[^\w\s&.,\-/]', '', dealer_name)
        
        return dealer_name.strip()
    
    def _is_valid_dealer_name(self, name: str) -> bool:
        """Check if dealer name passes basic validation."""
        if not name or len(name) < 3:
            return False
        
        # Should have at least some letters
        if not any(c.isalpha() for c in name):
            return False
        
        # Should not be just numbers
        if name.isdigit():
            return False
        
        # Check for common invalid patterns
        invalid_starts = ['total', 'amount', 'price', 'date', 'invoice', 'rs', '₹']
        if any(name.lower().startswith(invalid) for invalid in invalid_starts):
            return False
        
        # Should not be too long (likely a paragraph)
        if len(name) > 150:
            return False
        
        return True
    
    def _clean_and_validate_model_name(self, model_name: str) -> str:
        """Clean and validate model name."""
        if not model_name:
            return ""
        
        # Clean the name
        cleaned_name = self._clean_model_name(model_name)
        
        # Validate
        if not self._is_valid_model_name(cleaned_name):
            return ""
        
        return cleaned_name
    
    def _is_valid_model_name(self, model: str) -> bool:
        """Check if model name is valid."""
        if not model or len(model) < 2:
            return False
        
        # Should have at least one letter
        if not any(c.isalpha() for c in model):
            return False
        
        # Should not be just a number
        if model.isdigit():
            return False
        
        # Check for invalid patterns
        invalid_patterns = [
            r'^(?:TOTAL|AMOUNT|PRICE|DATE|DEALER|SELLER|RS|₹)$',
            r'^\d{1,2}$',  # Likely HP
        ]
        
        import re
        for pattern in invalid_patterns:
            if re.search(pattern, model, re.IGNORECASE):
                return False
        
        return True
    
    def _validate_signature(self, signature_result: Dict) -> Dict:
        """Validate signature detection result."""
        present = signature_result.get("present", False)
        confidence = signature_result.get("confidence", 0.0)
        bbox = signature_result.get("bbox", [])
        
        # Only consider signature present if confidence is high enough
        validated_present = present and confidence >= 0.6
        
        return {
            "present": validated_present,
            "bbox": bbox if validated_present else [],
            "confidence": confidence
        }
    
    def _validate_stamp(self, stamp_result: Dict) -> Dict:
        """Validate stamp detection result."""
        present = stamp_result.get("present", False)
        confidence = stamp_result.get("confidence", 0.0)
        bbox = stamp_result.get("bbox", [])
        
        # Only consider stamp present if confidence is high enough
        validated_present = present and confidence >= 0.5
        
        return {
            "present": validated_present,
            "bbox": bbox if validated_present else [],
            "confidence": confidence
        }
    
    def _clean_model_name(self, model_name: str) -> str:
        """Clean and standardize model name."""
        if not model_name:
            return ""
        
        # Convert to uppercase for consistency
        model_name = model_name.upper()
        
        # Remove common prefixes
        prefixes = ["MODEL:", "TRACTOR:", "DESCRIPTION:", "ITEM:"]
        for prefix in prefixes:
            if model_name.startswith(prefix):
                model_name = model_name[len(prefix):].strip()
        
        # Remove extra whitespace and special characters
        model_name = re.sub(r'\s+', ' ', model_name)
        model_name = re.sub(r'[^\w\s\-/]', '', model_name)
        
        return model_name.strip()
    
    def _validate_horse_power(self, horse_power: int) -> int:
        """Validate horse power value."""
        if not isinstance(horse_power, (int, float)):
            try:
                horse_power = int(horse_power)
            except:
                return 0
        
        # Ensure reasonable range
        if horse_power < 10 or horse_power > 200:
            # Try to find a reasonable default based on model
            return 50  # Common tractor HP
        else:
            return int(horse_power)
    
    def _validate_asset_cost(self, asset_cost: int) -> int:
        """Validate asset cost value."""
        if not isinstance(asset_cost, (int, float)):
            try:
                asset_cost = int(asset_cost)
            except:
                return 0
        
        # Ensure reasonable range for tractors
        if asset_cost < 10000:
            return 0
        elif asset_cost > 10000000:  # 1 crore
            # Might be in wrong units
            return int(asset_cost / 100000)  # Convert lakhs to rupees
        
        return int(asset_cost)
    
    def check_logical_consistency(self, fields: Dict) -> List[str]:
        """Check logical consistency between fields."""
        warnings = []
        
        # Check if horse power seems reasonable for the cost
        hp = fields.get("horse_power", 0)
        cost = fields.get("asset_cost", 0)
        
        if hp > 0 and cost > 0:
            # Typical cost per HP for tractors: ~10,000-20,000 Rs per HP
            cost_per_hp = cost / hp if hp > 0 else 0
            
            if cost_per_hp < 5000:
                warnings.append(f"Cost per HP ({cost_per_hp:.0f}) seems low")
            elif cost_per_hp > 50000:
                warnings.append(f"Cost per HP ({cost_per_hp:.0f}) seems high")
        
        # Check if dealer name contains company indicators
        dealer = fields.get("dealer_name", "")
        company_indicators = ["LTD", "PVT", "CORP", "COMPANY", "CO"]
        if dealer and not any(indicator in dealer.upper() for indicator in company_indicators):
            warnings.append("Dealer name may be incomplete")
        
        return warnings
    
    def fuzzy_match_dealer_name(self, extracted_name: str, master_list: List[str]) -> str:
        """Perform fuzzy matching with master dealer list."""
        if not extracted_name or not master_list:
            return extracted_name
        
        best_match = None
        best_score = 0
        
        for master_name in master_list:
            score = fuzz.ratio(extracted_name.lower(), master_name.lower())
            if score > best_score:
                best_score = score
                best_match = master_name
        
        if best_score >= 90:  # 90% match threshold
            return best_match
        else:
            return extracted_name