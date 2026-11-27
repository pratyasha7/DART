import re
import urllib.request
import os
from PIL import Image

def export_diagram(mermaid_code, base_name):
    out_dir = os.path.dirname(__file__)
    png_path = os.path.join(out_dir, f"{base_name}.png")
    svg_path = os.path.join(out_dir, f"{base_name}.svg")

    # Match flowchart directly, skipping any preceding init blocks
    m_flow = re.search(r'(flowchart\s+.*)', mermaid_code, re.DOTALL)
    if not m_flow:
        print(f"[SKIP] No flowchart block found for {base_name}")
        return
    clean_code = m_flow.group(1).strip().encode('utf-8')

    headers = {'Content-Type': 'text/plain; charset=utf-8', 'User-Agent': 'Mozilla/5.0'}

    # 1. Export SVG via Kroki
    print(f"Fetching {base_name} SVG via Kroki...")
    req_svg = urllib.request.Request('https://kroki.io/mermaid/svg', data=clean_code, headers=headers)
    try:
        with urllib.request.urlopen(req_svg, timeout=30) as resp:
            data_svg = resp.read()
        with open(svg_path, 'wb') as f:
            f.write(data_svg)
        print(f"[SUCCESS] Exported {svg_path} ({len(data_svg)/1024:.1f} KB)")
    except urllib.error.HTTPError as e:
        print(f"[ERROR] SVG HTTPError for {base_name}: {e.code} - {e.read().decode('utf-8')[:200]}")
    except Exception as e:
        print(f"[ERROR] SVG export failed for {base_name}: {e}")

    # 2. Export PNG via Kroki
    print(f"Fetching {base_name} PNG via Kroki...")
    req_png = urllib.request.Request('https://kroki.io/mermaid/png', data=clean_code, headers=headers)
    try:
        with urllib.request.urlopen(req_png, timeout=30) as resp:
            data_png = resp.read()
        with open(png_path, 'wb') as f:
            f.write(data_png)
        
        # Verify and resave true PNG
        im = Image.open(png_path)
        im.save(png_path, 'PNG')
        print(f"[SUCCESS] Exported {png_path} ({os.path.getsize(png_path)/1024:.1f} KB, dimensions: {im.size})")
    except urllib.error.HTTPError as e:
        print(f"[ERROR] PNG HTTPError for {base_name}: {e.code} - {e.read().decode('utf-8')[:200]}")
    except Exception as e:
        print(f"[ERROR] PNG export failed for {base_name}: {e}")

def export_all():
    doc_dir = os.path.dirname(__file__)
    
    diagrams = [
        ("ml_pipeline_diagram.html", "ml_pipeline_diagram"),
        ("implemented_architecture_diagram.html", "implemented_architecture_diagram"),
        ("simple_architecture_diagram.html", "simple_architecture_diagram"),
    ]

    for html_file, base_name in diagrams:
        full_path = os.path.join(doc_dir, html_file)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            m = re.search(r'<div class="mermaid">\s*(.*?)\s*</div>', content, re.DOTALL)
            if m:
                export_diagram(m.group(1).strip(), base_name)

if __name__ == "__main__":
    export_all()
