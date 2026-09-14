"""
scratch/report_generation/qc_audit.py
=====================================
"""
import pypdf

reader = pypdf.PdfReader('PROJECT_FINAL_INTEGRATED_VALIDATION_REPORT.pdf')
print(f'Total Pages: {len(reader.pages)}')

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    print(f"--- PAGE {i+1} ({len(lines)} lines) ---")
    for l in lines[:4]:
        print("  ", l[:90])

full_text = ' '.join(page.extract_text() for page in reader.pages)

prohibited = [
    '6.2 kHz', 'provably free', 'error-free', 'full 50',
    'zero directional ambiguities', 'complete mitigation', 'eliminated',
    'mechanical pitch tilt', '10,201', 'proven spatial audio-visual fidelity',
    'full horizontal and vertical operational fields', 'remarkable fidelity'
]
print("\n=== PROHIBITED PHRASES AUDIT ===")
all_p_pass = True
for p in prohibited:
    found = p.lower() in full_text.lower()
    status = 'FAIL' if found else 'PASS'
    if found:
        all_p_pass = False
    print(f"  [{status}] '{p}'")

required_numbers = [
    '7,676', '4.08 kHz', '0.25 deg', '107.5 deg', '90.8 deg', '65.0 deg',
    '+12.25 deg', '-6.0 deg', '-17.0 deg', '28.5 ms', '77.3 px', '187 frames'
]
print("\n=== REQUIRED TECHNICAL NUMBERS AUDIT ===")
all_r_pass = True
for r in required_numbers:
    found = r in full_text
    status = 'PASS' if found else 'FAIL'
    if not found:
        all_r_pass = False
    print(f"  [{status}] '{r}'")

print(f"\nAUDIT SUMMARY:")
print(f"  Pages: {len(reader.pages)}")
print(f"  Prohibited Phrases Check: {'ALL PASSED (0 found)' if all_p_pass else 'FAILED'}")
print(f"  Required Numbers Check: {'ALL PASSED (12/12 verified)' if all_r_pass else 'FAILED'}")

