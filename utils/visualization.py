import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

class Visualizer:
    """Visualize extraction results on documents."""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.OUTPUT_DIR) / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_visualization(self, 
                          image: np.ndarray,
                          ocr_result: Dict,
                          fields: Dict,
                          original_path: str):
        """Save visualization of extraction results."""
        # Create visualization image
        vis_image = self._create_visualization(image, ocr_result, fields)
        
        # Save to file
        output_path = self.output_dir / f"{Path(original_path).stem}_vis.png"
        cv2.imwrite(str(output_path), vis_image)
        
        # Create detailed visualization with text
        self._save_detailed_visualization(image, ocr_result, fields, original_path)
    
    def _create_visualization(self, 
                            image: np.ndarray,
                            ocr_result: Dict,
                            fields: Dict) -> np.ndarray:
        """Create visualization with bounding boxes."""
        # Create copy of image
        vis = image.copy()
        
        # Draw OCR bounding boxes
        for i, (text, bbox) in enumerate(zip(ocr_result.get("text", []), 
                                           ocr_result.get("boxes", []))):
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw signature bounding box
        if fields.get("signature", {}).get("present", False):
            bbox = fields["signature"]["bbox"]
            if len(bbox) == 4:
                cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
                cv2.putText(vis, "Signature", (bbox[0], bbox[1]-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw stamp bounding box
        if fields.get("stamp", {}).get("present", False):
            bbox = fields["stamp"]["bbox"]
            if len(bbox) == 4:
                cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3)
                cv2.putText(vis, "Stamp", (bbox[0], bbox[1]-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Add extracted information text overlay
        self._add_text_overlay(vis, fields)
        
        return vis
    
    def _add_text_overlay(self, image: np.ndarray, fields: Dict):
        """Add extracted fields as text overlay."""
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        bg_color = (0, 0, 0)
        thickness = 2
        
        # Create text lines
        lines = [
            f"Dealer: {fields.get('dealer_name', 'N/A')}",
            f"Model: {fields.get('model_name', 'N/A')}",
            f"HP: {fields.get('horse_power', 'N/A')}",
            f"Cost: {fields.get('asset_cost', 'N/A'):,}",
            f"Signature: {'Yes' if fields.get('signature', {}).get('present') else 'No'}",
            f"Stamp: {'Yes' if fields.get('stamp', {}).get('present') else 'No'}"
        ]
        
        # Add text with background
        for line in lines:
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            
            # Draw background rectangle
            cv2.rectangle(image, (10, y_offset - 25), 
                         (10 + text_size[0] + 10, y_offset + 5), 
                         bg_color, -1)
            
            # Draw text
            cv2.putText(image, line, (15, y_offset), 
                       font, font_scale, color, thickness)
            
            y_offset += 30
    
    def _save_detailed_visualization(self, 
                                   image: np.ndarray,
                                   ocr_result: Dict,
                                   fields: Dict,
                                   original_path: str):
        """Save detailed visualization with matplotlib."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot original image with boxes
        ax1.imshow(image_rgb)
        ax1.set_title('Original Image with Detections')
        ax1.axis('off')
        
        # Draw bounding boxes
        self._plot_ocr_boxes(ax1, ocr_result)
        self._plot_field_boxes(ax1, fields)
        
        # Create field information display
        self._plot_field_info(ax2, fields, ocr_result)
        
        # Save figure
        output_path = self.output_dir / f"{Path(original_path).stem}_detailed.png"
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_ocr_boxes(self, ax, ocr_result: Dict):
        """Plot OCR bounding boxes."""
        for i, (text, bbox) in enumerate(zip(ocr_result.get("text", []), 
                                           ocr_result.get("boxes", []))):
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=1, edgecolor='green', 
                                       facecolor='none', alpha=0.5)
                ax.add_patch(rect)
                
                # Add text label (for first few boxes to avoid clutter)
                if i < 10:
                    ax.text(x1, y1-5, text[:20], fontsize=8, 
                           color='green', backgroundcolor='white')
    
    def _plot_field_boxes(self, ax, fields: Dict):
        """Plot field-specific bounding boxes."""
        # Signature box
        if fields.get("signature", {}).get("present", False):
            bbox = fields["signature"]["bbox"]
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=3, edgecolor='red', 
                                       facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-10, "SIGNATURE", fontsize=10, 
                       color='red', fontweight='bold')
        
        # Stamp box
        if fields.get("stamp", {}).get("present", False):
            bbox = fields["stamp"]["bbox"]
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=3, edgecolor='blue', 
                                       facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-10, "STAMP", fontsize=10, 
                       color='blue', fontweight='bold')
    
    def _plot_field_info(self, ax, fields: Dict, ocr_result: Dict):
        """Plot field information."""
        ax.axis('off')
        ax.set_title('Extracted Information', fontsize=16, fontweight='bold')
        
        info_text = []
        info_text.append("EXTRACTED FIELDS:")
        info_text.append("=" * 30)
        
        # Add fields
        info_text.append(f"Dealer Name: {fields.get('dealer_name', 'N/A')}")
        info_text.append(f"Model Name: {fields.get('model_name', 'N/A')}")
        info_text.append(f"Horse Power: {fields.get('horse_power', 'N/A')}")
        info_text.append(f"Asset Cost: {fields.get('asset_cost', 'N/A'):,}")
        info_text.append(f"Signature Detected: {'Yes' if fields.get('signature', {}).get('present') else 'No'}")
        info_text.append(f"Stamp Detected: {'Yes' if fields.get('stamp', {}).get('present') else 'No'}")
        
        info_text.append("")
        info_text.append("OCR STATISTICS:")
        info_text.append("=" * 30)
        info_text.append(f"Total Text Blocks: {len(ocr_result.get('text', []))}")
        info_text.append(f"Avg Confidence: {ocr_result.get('avg_confidence', 0):.2%}")
        info_text.append(f"Engine: {ocr_result.get('engine', 'N/A')}")
        
        # Display as text
        ax.text(0.1, 0.95, '\n'.join(info_text), 
               fontsize=10, family='monospace',
               verticalalignment='top',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))