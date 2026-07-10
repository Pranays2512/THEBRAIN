import fitz  # PyMuPDF
import json
import argparse
import sys
from pathlib import Path

def extract_structured_text(pdf_path: str) -> dict:
    """
    Extracts text from a PDF in a structured way using PyMuPDF.
    Returns a dictionary containing pages, blocks, lines, and spans.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}", file=sys.stderr)
        return None

    structured_data = {
        "file": pdf_path,
        "metadata": doc.metadata,
        "pages": []
    }

    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 'dict' format returns a detailed dictionary including images and text
        # with bounding boxes, font information, and reading order.
        page_dict = page.get_text("dict")
        
        page_data = {
            "page_number": page_num + 1,
            "width": page_dict["width"],
            "height": page_dict["height"],
            "blocks": []
        }

        # Blocks can be text (type 0) or image (type 1)
        for block in page_dict.get("blocks", []):
            if block["type"] == 0:  # Text block
                block_data = {
                    "bbox": block["bbox"],  # [x0, y0, x1, y1]
                    "lines": []
                }
                
                for line in block.get("lines", []):
                    line_data = {
                        "bbox": line["bbox"],
                        "spans": []
                    }
                    
                    for span in line.get("spans", []):
                        span_data = {
                            "text": span["text"],
                            "font": span["font"],
                            "size": span["size"],
                            "color": span["color"],
                            "bbox": span["bbox"]
                        }
                        line_data["spans"].append(span_data)
                        
                    block_data["lines"].append(line_data)
                    
                page_data["blocks"].append(block_data)
                
        structured_data["pages"].append(page_data)

    doc.close()
    return structured_data

def main():
    parser = argparse.ArgumentParser(description="Extract structured text from a PDF.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("-o", "--output", help="Path to save the JSON output", default=None)
    parser.add_argument("-p", "--pretty", action="store_true", help="Pretty print JSON")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_path)
    if not pdf_path.is_file():
        print(f"Error: File '{args.pdf_path}' not found.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Processing '{args.pdf_path}'...", file=sys.stderr)
    data = extract_structured_text(str(pdf_path))
    
    if data is None:
        sys.exit(1)
        
    indent = 4 if args.pretty else None
    json_output = json.dumps(data, indent=indent, ensure_ascii=False)
    
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_output)
            print(f"Structured data saved to '{args.output}'", file=sys.stderr)
        except Exception as e:
            print(f"Error saving to {args.output}: {e}", file=sys.stderr)
    else:
        print(json_output)

if __name__ == "__main__":
    main()
