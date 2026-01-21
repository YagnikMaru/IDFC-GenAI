# Intelligent Document AI for Field Extraction

## Overview
An end-to-end Document AI pipeline for extracting key fields from invoices and quotations, specifically designed for tractor loan documents but generalizable to any invoice type. The system handles multiple languages (English, Hindi, Gujarati), various document qualities (scanned, digital, handwritten), and extracts structured data with high accuracy.

## Features
- **Multilingual OCR**: Supports English, Hindi, and Gujarati
- **Field Extraction**: Dealer name, model name, horse power, asset cost
- **Visual Detection**: Signature and stamp detection with bounding boxes
- **Cost Optimization**: Low-cost inference design (<$0.01 per document)
- **High Accuracy**: Target ≥95% document-level accuracy
- **Explainability**: Confidence scores and validation checks
- **Scalable Architecture**: Modular design for easy maintenance

## Architecture
```
┌─────────────────────────────────────────────────────────┐
│ Document Ingestion │
│ (PDF/Image → Preprocessing) │
└───────────────────────────┬─────────────────────────────┘
│
┌───────────────────────────▼─────────────────────────────┐
│ OCR & Layout Analysis │
│ (PaddleOCR/Tesseract + Text Structuring) │
└───────────────────────────┬─────────────────────────────┘
│
┌───────────────────────────▼─────────────────────────────┐
│ Field Extraction │
│ (Regex + NLP + Fuzzy Matching + Master Validation) │
└───────────────────────────┬─────────────────────────────┘
│
┌───────────────────────────▼─────────────────────────────┐
│ Signature & Stamp Detection │
│ (YOLO/CNN + Template Matching) │
└───────────────────────────┬─────────────────────────────┘
│
┌───────────────────────────▼─────────────────────────────┐
│ Post-processing │
│ (Validation + Confidence Scoring + Formatting) │
└───────────────────────────┬─────────────────────────────┘
│
┌───────────────────────────▼─────────────────────────────┐
│ JSON Output │
│ (Structured Data + Metadata + Confidence Scores) │
└─────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd newcode
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PaddleOCR dependencies**
   ```bash
   python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
   ```

5. **Download NLP models**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Single Document Processing
```bash
python main.py --single path/to/document.png
```

### Batch Processing
```bash
python main.py --batch path/to/documents/folder
```

### Configuration
Edit `config.py` to customize:
- OCR engines (PaddleOCR, Tesseract, EasyOCR)
- Field extraction rules
- Confidence thresholds
- Output formats

## Project Structure
```
newcode/
├── main.py                 # Main entry point
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── utils/
│   ├── field_extractor_final.py    # Field extraction logic
│   ├── ocr_engine.py       # OCR processing
│   ├── signature_detector_final.py # Signature detection
│   └── visualization.py    # Result visualization
├── train/                  # Training data (images)
├── outputs/                # Generated results
└── tests/                  # Test files
```

## Testing
Run the test suite:
```bash
python -m pytest tests/
```

## Output Format
The system outputs JSON files with the following structure:
```json
{
  "document_id": "filename",
  "extracted_fields": {
    "dealer_name": "ABC Tractors",
    "model_name": "Mahindra 575 DI",
    "horse_power": "47 HP",
    "asset_cost": "₹5,50,000"
  },
  "visual_detections": {
    "signatures": [{"bbox": [x1,y1,x2,y2], "confidence": 0.95}],
    "stamps": [{"bbox": [x1,y1,x2,y2], "confidence": 0.89}]
  },
  "confidence_scores": {
    "overall": 0.92,
    "field_accuracy": 0.95
  },
  "processing_metadata": {
    "ocr_engine": "paddleocr",
    "processing_time": 2.3
  }
}
```

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Support
For questions or issues, please open an issue on GitHub or contact the development team.