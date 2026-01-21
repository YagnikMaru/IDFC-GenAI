import os
import json
import time
import uuid
import sys
from typing import Dict, Any
from datetime import datetime
import cv2
import numpy as np
from pathlib import Path

# Add the parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.ocr_engine import OCREngine
from utils.field_extractor import FieldExtractor
from utils.signature_detector_final import SignatureDetector
from utils.post_processor import PostProcessor

class InvoiceExtractionSystem:
    """Main system for extracting fields from invoice documents."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.ocr_engine = OCREngine(self.config)
        self.field_extractor = FieldExtractor(self.config)
        self.signature_detector = SignatureDetector(self.config)
        self.post_processor = PostProcessor(self.config)
        
    def process_document(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single document and extract all required fields.
        
        Args:
            image_path: Path to the document image
            
        Returns:
            Dictionary containing extracted fields and metadata
        """
        start_time = time.time()
        
        try:
            print(f"Processing: {image_path}")
            
            # Step 1: Read image
            image = cv2.imread(image_path)
            if image is None:
                # Try alternative method for PDFs
                if image_path.lower().endswith('.pdf'):
                    from pdf2image import convert_from_path
                    images = convert_from_path(image_path)
                    if images:
                        image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
                    else:
                        raise ValueError(f"Could not read PDF: {image_path}")
                else:
                    raise ValueError(f"Could not read image: {image_path}")
            
            # Step 2: Preprocess
            preprocessed = self.ocr_engine.preprocess_image(image)
            
            # Step 3: Perform OCR
            ocr_result = self.ocr_engine.extract_text_and_layout(preprocessed)
            
            # Step 4: Extract fields
            text_fields = self.field_extractor.extract_fields(ocr_result)
            
            # Step 5: Detect signatures and stamps
            signature_result = self.signature_detector.detect_signature(image)
            stamp_result = self.signature_detector.detect_stamp(image)
            
            # Step 6: Post-process
            processed_fields = self.post_processor.validate_and_process(
                text_fields, signature_result, stamp_result
            )
            
            # Step 7: Prepare output
            processing_time = time.time() - start_time
            cost_estimate = self._estimate_cost(processing_time)
            confidence = self._calculate_confidence(processed_fields, ocr_result)
            
            # Generate doc_id from filename
            doc_id = Path(image_path).stem
            doc_id = doc_id.replace(' ', '_').replace('-', '_')
            
            result = {
                "doc_id": f"{doc_id}_{uuid.uuid4().hex[:4]}",
                "fields": processed_fields,
                "confidence": confidence,
                "processing_time_sec": round(processing_time, 2),
                "cost_estimate_usd": cost_estimate
            }
            
            print(f"  Completed in {processing_time:.2f}s, Confidence: {confidence:.2f}")
            return result
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return self._get_error_response(image_path, str(e))
    
    def process_batch(self, input_dir: str, output_dir: str = None) -> None:
        """
        Process all documents in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results (default: sample_output/)
        """
        if output_dir is None:
            output_dir = self.config.OUTPUT_DIR
        
        results = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf']
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Processing batch from: {input_dir}")
        print(f"Output directory: {output_dir}")
        
        # Get all files
        files = list(Path(input_dir).iterdir())
        total_files = len([f for f in files if f.suffix.lower() in image_extensions])
        
        print(f"Found {total_files} documents to process")
        
        # Process all images in the directory
        processed_count = 0
        for file_path in files:
            if file_path.suffix.lower() in image_extensions:
                processed_count += 1
                print(f"\n[{processed_count}/{total_files}] ", end="")
                
                result = self.process_document(str(file_path))
                results.append(result)
                
                # Save individual result
                output_file = Path(output_dir) / f"{file_path.stem}_result.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save combined results
        if results:
            combined_output = {
                "timestamp": datetime.now().isoformat(),
                "total_documents": len(results),
                "successful_documents": len([r for r in results if not r.get("error", False)]),
                "average_confidence": np.mean([r["confidence"] for r in results if not r.get("error", False)] or [0]),
                "average_processing_time": np.mean([r["processing_time_sec"] for r in results if not r.get("error", False)] or [0]),
                "results": results
            }
            
            combined_file = Path(output_dir) / "combined_results.json"
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(combined_output, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*50}")
        print(f"Batch processing complete!")
        print(f"Processed: {len(results)} documents")
        if results:
            successful = len([r for r in results if not r.get("error", False)])
            print(f"Successful: {successful} documents")
            avg_conf = combined_output['average_confidence']
            print(f"Average confidence: {avg_conf:.2%}")
        print(f"Results saved to: {output_dir}")
    
    def _calculate_confidence(self, fields: Dict, ocr_result: Dict) -> float:
        """Calculate overall confidence score."""
        scores = []
        
        # Field presence score
        required_fields = ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
        present_fields = sum(1 for field in required_fields if fields.get(field))
        if present_fields > 0:
            scores.append(present_fields / len(required_fields))
        
        # OCR quality score
        if ocr_result.get('avg_confidence'):
            scores.append(ocr_result['avg_confidence'])
        
        # Signature/stamp detection confidence
        sig_conf = fields.get('signature', {}).get('confidence', 0)
        if fields.get('signature', {}).get('present') and sig_conf > 0:
            scores.append(sig_conf)
        
        stamp_conf = fields.get('stamp', {}).get('confidence', 0)
        if fields.get('stamp', {}).get('present') and stamp_conf > 0:
            scores.append(stamp_conf)
        
        return round(np.mean(scores), 2) if scores else 0.0
    
    def _estimate_cost(self, processing_time: float) -> float:
        """Estimate cost based on processing time."""
        # Conservative estimate: $0.05 per hour for CPU
        hourly_rate = 0.05
        cost = (processing_time / 3600) * hourly_rate
        return round(max(cost, 0.0001), 4)  # Minimum cost
    
    def _get_error_response(self, image_path: str, error_msg: str = "") -> Dict:
        """Return error response for failed processing."""
        doc_id = Path(image_path).stem
        doc_id = doc_id.replace(' ', '_').replace('-', '_')
        
        return {
            "doc_id": f"{doc_id}_error",
            "fields": {
                "dealer_name": "",
                "model_name": "",
                "horse_power": 0,
                "asset_cost": 0,
                "signature": {"present": False, "bbox": []},
                "stamp": {"present": False, "bbox": []}
            },
            "confidence": 0.0,
            "processing_time_sec": 0.0,
            "cost_estimate_usd": 0.0,
            "error": True,
            "error_message": error_msg
        }


def main():
    """Main execution function."""
    # Initialize system
    config = Config()
    system = InvoiceExtractionSystem(config)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Process single file
        if sys.argv[1] == "--single":
            if len(sys.argv) > 2:
                result = system.process_document(sys.argv[2])
                print("\n" + "="*50)
                print("EXTRACTION RESULT:")
                print("="*50)
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print("Usage: python main.py --single <image_path>")
        
        # Process batch
        elif sys.argv[1] == "--batch":
            input_dir = sys.argv[2] if len(sys.argv) > 2 else config.TRAIN_DIR
            output_dir = sys.argv[3] if len(sys.argv) > 3 else config.OUTPUT_DIR
            system.process_batch(input_dir, output_dir)
        
        # Test with sample image
        elif sys.argv[1] == "--test":
            # Use a sample image if available
            sample_images = list(Path(config.TRAIN_DIR).glob("*.jpg")) + \
                          list(Path(config.TRAIN_DIR).glob("*.png"))
            if sample_images:
                result = system.process_document(str(sample_images[0]))
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print("No sample images found in train/ directory")
        
        else:
            print("Usage:")
            print("  Single file: python main.py --single <image_path>")
            print("  Batch:       python main.py --batch [input_dir] [output_dir]")
            print("  Test:        python main.py --test")
    else:
        # Default: process all images in train directory
        print("No arguments provided. Running batch processing on train/ directory...")
        system.process_batch(config.TRAIN_DIR, config.OUTPUT_DIR)


if __name__ == "__main__":
    main()