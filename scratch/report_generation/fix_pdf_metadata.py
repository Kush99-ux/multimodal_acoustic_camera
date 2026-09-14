from pypdf import PdfReader, PdfWriter

input_file = "PROJECT_FINAL_INTEGRATED_VALIDATION_REPORT.pdf"
output_file = "PROJECT_FINAL_INTEGRATED_VALIDATION_REPORT_fixed.pdf"

reader = PdfReader(input_file)
writer = PdfWriter()

for page in reader.pages:
    writer.add_page(page)

writer.add_metadata({
    "/Title": "Real-Time Multimodal Acoustic Camera",
    "/Author": "Kush Sahu",
    "/Subject": "Design, Multi-Band MUSIC Localization, Vision Synchronization, and Hardware Validation",
})

with open(output_file, "wb") as f:
    writer.write(f)

print(f"Created: {output_file}")