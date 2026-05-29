# Paddle OCR Project

A Python-based OCR (Optical Character Recognition) solution using PaddleOCR-VL (Vision Language) model for document text extraction with post-processing capabilities.

## Project Structure

- **OCR.py** - Main OCR processing script that loads the PaddleOCR-VL model and extracts text from images/PDFs
- **post_processing.py** - Post-processing module for cleaning, normalizing, and refining extracted text
- **requirements.txt** - All project dependencies
- **document_folder/** - Directory to place your input images/PDFs for processing

## Setup Instructions

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager

### 2. Clone/Download the Project
```bash
cd Paddle_OCR
```

### 3. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Prepare Your Documents
- Place your image or PDF files in the `document_folder/` directory
- Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`, `.pdf`

### 6. Run OCR Processing
```bash
python OCR.py
```

This will:
- Load the PaddleOCR-VL model
- Process files from `document_folder/`
- Save raw OCR results to `raw_result/` directory

### 7. Post-process Results 
```bash
# Process a single JSON result file
python post_processing.py --input result_1.json

# Process and save to custom output
python post_processing.py --input result_1.json --output cleaned.txt

# Process entire directory
python post_processing.py --input ./raw_result/ --output ./cleaned/
```

## GPU Setup (CUDA)

If you have an NVIDIA GPU and want to use GPU acceleration instead of CPU:

### 1. Install CUDA Toolkit
- Download and install [NVIDIA CUDA Toolkit 11.8+](https://developer.nvidia.com/cuda-toolkit)
- Download and install [cuDNN](https://developer.nvidia.com/cudnn)

### 2. Update Dependencies for GPU
```bash
# Uninstall CPU version if already installed
pip uninstall paddlepaddle -y

# Install GPU version
pip install paddlepaddle-gpu
```

### 3. Modify OCR.py
In `OCR.py`, change the `DEVICE` variable from:
```python
DEVICE = "cpu"
```
to:
```python
DEVICE = "gpu"
```

### 4. Run with GPU
```bash
python OCR.py
```

The model will automatically use your GPU for faster processing. GPU processing can be 5-10x faster than CPU depending on your hardware.

### 5. Troubleshooting GPU Issues
- Verify CUDA installation: `nvidia-smi`
- Check PaddlePaddle GPU support: `python -c "import paddle; print(paddle.device.get_device())"`
- Ensure GPU memory is sufficient (minimum 4GB VRAM recommended)

## Performance Notes

- **CPU Mode**: Works on any system but slower
- **GPU Mode**: Requires NVIDIA GPU with CUDA support, significantly faster
- You can adjust `OMP_NUM_THREADS` in `OCR.py` based on your CPU cores for optimization

## Output

- **Raw OCR Results**: Saved in `raw_result/` directory (JSON format)
- **Cleaned Results**: After post-processing, saved to specified output directory

---

For questions or issues, refer to [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
