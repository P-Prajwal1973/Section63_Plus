import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, Image)

styles = getSampleStyleSheet()
small = styles["Normal"].clone("small", fontSize=8)


def make_table(fields: dict):
    rows = [[Paragraph(f"<b>{k}</b>", small), Paragraph(str(v), small)]
            for k, v in fields.items()]
    t = Table(rows, colWidths=[140, 310])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def generate_certificate(data: dict, output_path="certificate.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    s = []
    s.append(Paragraph("CERTIFICATE UNDER SECTION 63(4)", styles["Title"]))
    s.append(Paragraph("Bharatiya Sakshya Adhiniyam, 2023 (DRAFT)", styles["Normal"]))
    s.append(Spacer(1, 12))

    s.append(Paragraph("PART A", styles["Heading2"]))
    s.append(make_table(data["part_a"]))
    s.append(Spacer(1, 12))

    s.append(Paragraph("PART B", styles["Heading2"]))
    s.append(make_table(data["part_b"]))
    s.append(Spacer(1, 12))

    s.append(Paragraph("AI-GENERATION RISK ANNEXURE", styles["Heading2"]))
    s.append(make_table(data["ai_annexure"]))
    heatmap = data.get("heatmap_path")
    if heatmap and os.path.exists(heatmap):
        s.append(Spacer(1, 8))
        s.append(Image(heatmap, width=250, height=250))

    s.append(Spacer(1, 12))
    s.append(Paragraph("Chain-of-custody log", styles["Heading2"]))
    log_rows = {f"{e['timestamp']}": f"{e['action']} | {e['entry_hash'][:16]}..."
                for e in data.get("custody_entries", [])}
    if log_rows:
        s.append(make_table(log_rows))

    s.append(Spacer(1, 16))
    s.append(Paragraph(
        "This is a machine-drafted certificate. It must be reviewed and signed "
        "by the responsible officer/expert before submission.", small))
    doc.build(s)
    return output_path


if __name__ == "__main__":
    sample = {
        "part_a": {"Name": "Test Officer", "Designation": "Inspector",
                   "Device description": "Samsung phone", "Manner of production": "Screen recording"},
        "part_b": {"Expert": "Test Expert", "Hash algorithm": "SHA-256",
                   "Hash value": "a3f5" * 16},
        "ai_annexure": {"Risk score": "0.91 (High)", "Model": "EfficientNet-B0",
                        "Dataset": "FaceForensics++ C23"},
    }
    print("Created:", generate_certificate(sample))