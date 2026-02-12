"""PDF processing utilities for loading and extracting text from PDF files."""

import os
import io
import re
import glob
import fitz
import pdfplumber
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
import sys
from pathlib import Path

# Add config to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'config'))
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def is_likely_garbled_pdf_text(text):
    """Check if PDF text is likely garbled or unreadable."""
    if not text or len(text) < 100:
        return True
    alpha_count = sum(c.isalnum() for c in text)
    non_alpha_count = sum(not c.isalnum() for c in text)
    ratio = alpha_count / non_alpha_count if non_alpha_count else 1.0
    if ratio < 0.1:
        return True
    control_chars = len(re.findall(r"[\x00-\x1F\x7F]", text))
    if control_chars > 10:
        return True
    return False


def extract_text_from_page(page, file_path, page_num):
    """Extract text from a PDF page using multiple fallback methods."""
    # Try PyMuPDF first
    pdf_text = page.get_text()
    
    if is_likely_garbled_pdf_text(pdf_text):
        print("Switched to pdfplumber")
        plumber_doc = pdfplumber.open(file_path)
        pdf_text = plumber_doc.pages[page_num].extract_text()
        
        if is_likely_garbled_pdf_text(pdf_text):
            try:
                print("Switched to pytesseract (OCR)")
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.pil_tobytes(format="PNG")
                img_obj = Image.open(io.BytesIO(img_bytes))
                pdf_text = pytesseract.image_to_string(img_obj)
            except Exception as e:
                print(f"Warning: OCR (tesseract) failed: {e}")
                print("Using garbled text from pdfplumber as fallback")
                # Keep the garbled text from pdfplumber as last resort
    
    return pdf_text


def load_pdf_documents(data_dir=None, file_paths=None):
    """
    Load and process PDF files from the specified directory or file paths.
    
    Args:
        data_dir: Directory containing PDF files (optional, defaults to DATA_DIR)
        file_paths: List of specific PDF file paths to process (optional, takes precedence over data_dir)
    
    Returns:
        List of Document objects
    """
    all_docs = []
    
    # If file_paths is provided, use those directly
    if file_paths is not None:
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        # Validate that all files exist
        valid_paths = []
        for path in file_paths:
            if os.path.isfile(path):
                valid_paths.append(path)
            else:
                print(f"WARNING: File not found: {path}")
        
        if not valid_paths:
            print("WARNING: No valid PDF files found in provided file_paths!")
            return all_docs
        
        print(f"Processing {len(valid_paths)} PDF file(s) from provided paths: {valid_paths}")
        file_paths = valid_paths
    else:
        # Default behavior: load from directory
        if data_dir is None:
            data_dir = str(DATA_DIR)
        
        pdf_files = glob.glob(os.path.join(data_dir, '*.pdf'))
        file_paths = pdf_files if pdf_files else []
        
        print(f"Looking for PDFs in: {data_dir}")
        print(f"Found {len(file_paths)} PDF file(s): {file_paths}")
        
        if not file_paths:
            print("WARNING: No PDF files found! Please ensure PDF files are in the data directory.")
            return all_docs
    
    # Process all PDF files
    print(f"Processing {len(file_paths)} PDF file(s)...")
    for file_path in file_paths:
        print(f"Processing: {file_path}")
        try:
            pdf_doc = fitz.open(file_path)
            for page in pdf_doc:
                page_num = page.number
                print(f"  Page {page_num}")
                pdf_text = extract_text_from_page(page, file_path, page_num)
                chunks = pdf_splitter.split_text(pdf_text)
                docs = [
                    Document(
                        page_content=chunk,
                        metadata={"page": page_num + 1, "source": file_path.split("/")[-1]}
                    )
                    for chunk in chunks
                ]
                all_docs.extend(docs)
            pdf_doc.close()
        except Exception as e:
            print(f"ERROR: Failed to process {file_path}: {e}")
            continue
    
    print(f"Total documents loaded: {len(all_docs)}")
    return all_docs
