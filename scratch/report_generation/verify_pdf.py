"""
scratch/report_generation/verify_pdf.py
=======================================
Quick verification utility to inspect the generated PDF page streams.
"""
import zlib
import re

with open('PROJECT_FINAL_INTEGRATED_VALIDATION_REPORT.pdf', 'rb') as f:
    data = f.read()

streams = re.findall(rb'stream[\r\n]+(.*?)[\r\n]+endstream', data, re.DOTALL)
print(f'Total streams found: {len(streams)}')

page_texts = []
for i, s in enumerate(streams):
    try:
        decomp = zlib.decompress(s)
        bt_blocks = re.findall(rb'BT(.*?)ET', decomp, re.DOTALL)
        if bt_blocks:
            text_chunks = []
            for b in bt_blocks:
                strings = re.findall(rb'\((.*?)\)', b)
                for st in strings:
                    try:
                        text_chunks.append(st.decode('latin1'))
                    except Exception:
                        pass
            page_text = ' '.join(text_chunks)
            if len(page_text) > 40:
                page_texts.append(page_text)
    except Exception:
        pass

print(f'Text-bearing streams found: {len(page_texts)}')
for idx, pt in enumerate(page_texts):
    print(f"Page {idx+1}: {pt[:120]}...")
