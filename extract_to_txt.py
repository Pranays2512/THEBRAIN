import fitz
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_to_txt.py <input.pdf> <output.txt>")
        sys.exit(1)
        
    input_pdf = sys.argv[1]
    output_txt = sys.argv[2]
    
    try:
        doc = fitz.open(input_pdf)
    except Exception as e:
        print(f"Error opening {input_pdf}: {e}", file=sys.stderr)
        sys.exit(1)
        
    with open(output_txt, "w", encoding="utf-8") as f:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            f.write(text)
            f.write("\n")
            
    print(f"Successfully extracted {len(doc)} pages to {output_txt}")

if __name__ == "__main__":
    main()
