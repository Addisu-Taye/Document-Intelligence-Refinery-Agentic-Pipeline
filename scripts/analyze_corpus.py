# scripts/analyze_corpus.py
import pdfplumber
from pathlib import Path

CORPUS_DIR = Path("corpus")
FILES = {
    "CBE (Native Digital)": CORPUS_DIR / "CBE ANNUAL REPORT 2023-24.pdf",
    "Audit (Scanned Image)": CORPUS_DIR / "Audit Report - 2023.pdf"
}

def analyze_page(page):
    """Extract metrics for a single page with robust error handling."""
    page_area = page.width * page.height
    
    # Safe char extraction
    chars = page.chars if hasattr(page, 'chars') else []
    char_count = len(chars)
    
    # Safe image extraction
    images = page.images if hasattr(page, 'images') else []
    image_count = len(images)
    image_area = sum(img.get('width', 0) * img.get('height', 0) for img in images)
    image_ratio = (image_area / page_area) if page_area > 0 else 0
    
    # Character density
    char_density = (char_count / page_area) if page_area > 0 else 0
    
    # Table detection
    try:
        tables = page.find_tables()
        table_count = len(tables)
    except:
        table_count = 0
    
    # Sample char info (safe access)
    sample_char_info = None
    if char_count > 0 and chars:
        first_char = chars[0]
        # Safely get bbox - keys might vary
        bbox = first_char.get('bbox') or first_char.get('x0', 'N/A')
        sample_char_info = f"bbox={bbox}, text='{first_char.get('text', '')[:10]}'"
    
    # Debug: What keys do chars actually have?
    char_keys_sample = None
    if char_count > 0 and chars:
        char_keys_sample = list(chars[0].keys())[:5]  # Show first 5 keys
    
    return {
        "page_num": page.page_number,
        "char_count": char_count,
        "char_density": round(char_density, 6),
        "image_count": image_count,
        "image_ratio": round(image_ratio, 4),
        "table_count": table_count,
        "sample_char_info": sample_char_info,
        "char_keys_sample": char_keys_sample
    }

def analyze_document(name, path):
    print(f"\n{'='*20} {name} {'='*20}")
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    with pdfplumber.open(path) as pdf:
        print(f"📄 Total Pages: {len(pdf.pages)}")
        print(f"📐 Page Size: {pdf.pages[0].width} x {pdf.pages[0].height} points")
        
        # Check for text layer at PDF level
        has_text_layer = any(len(page.chars) > 0 for page in pdf.pages[:5])
        print(f"🔤 Has Text Layer (first 5 pages): {has_text_layer}")
        
        print(f"\n{'Page':<5} | {'Chars':<8} | {'Density':<10} | {'Img Ratio':<10} | {'Tables':<6} | {'Sample Info'}")
        print("-" * 90)
        
        for page in pdf.pages[:3]:  # First 3 pages only
            metrics = analyze_page(page)
            sample_str = metrics['sample_char_info'] or "No chars"
            if metrics['char_keys_sample']:
                sample_str += f" [keys: {metrics['char_keys_sample']}]"
            
            print(f"{metrics['page_num']:<5} | {metrics['char_count']:<8} | {metrics['char_density']:<10} | {metrics['image_ratio']:<10} | {metrics['table_count']:<6} | {sample_str}")

if __name__ == "__main__":
    print("🔍 Starting Document Corpus Analysis...")
    for name, path in FILES.items():
        analyze_document(name, path)
    print("\n💡 KEY INSIGHT: If 'native' PDFs show 0 chars, they may be image-based!")