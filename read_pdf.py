import sys
import pypdf

def main():
    if len(sys.argv) < 2:
        print("Usage: python read_pdf.py <pdf_path>")
        return
        
    pdf_path = sys.argv[1]
    
    with open("pdf_content_fixed.txt", "w", encoding="utf-8") as f:
        reader = pypdf.PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            f.write(f"\n--- Page {i+1} ---\n")
            try:
                f.write(page.extract_text() + "\n")
            except Exception as e:
                f.write(f"Error extracting text from page {i+1}: {e}\n")

if __name__ == "__main__":
    main()
