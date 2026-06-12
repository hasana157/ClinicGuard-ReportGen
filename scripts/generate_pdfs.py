"""
Utility script to compile markdown reports into professional PDF files.

If reportlab is installed, creates reports/TECHNICAL_REPORT.pdf and
reports/HALLUCINATION_ANALYSIS.pdf. If not, outputs a warning and instructions.
"""

import os
import re
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import REPORTS_DIR


def generate_pdf_from_md(md_path: str, pdf_path: str):
    """
    Compiles a markdown file to a PDF file using ReportLab.
    """
    print(f"Compiling {md_path} into {pdf_path}...")
    
    if not os.path.exists(md_path):
        print(f"Error: source file {md_path} does not exist.")
        return
        
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        
        # Read markdown lines
        with open(md_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
        story = []
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            name="TitleStyle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#302b63"),
            spaceAfter=20
        )
        
        h2_style = ParagraphStyle(
            name="H2Style",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#24243e"),
            spaceBefore=12,
            spaceAfter=8
        )
        
        body_style = ParagraphStyle(
            name="BodyStyle",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#333333"),
            spaceAfter=8
        )
        
        table_text_style = ParagraphStyle(
            name="TableTextStyle",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#333333")
        )
        
        table_header_style = ParagraphStyle(
            name="TableHeaderStyle",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
            textColor=colors.white,
            fontName="Helvetica-Bold"
        )

        in_table = False
        table_data = []
        
        for line in lines:
            line_str = line.strip()
            
            # Handle table blocks
            if line_str.startswith("|"):
                # Parse columns
                cols = [c.strip() for c in line_str.split("|")[1:-1]]
                if not cols or all(not c for c in cols):
                    continue
                in_table = True
                # Ignore separator row e.g. |---|---|
                if any(c.startswith("-") for c in cols):
                    continue
                
                # Check header vs cell
                if not table_data:
                    # Header row
                    table_data.append([Paragraph(c, table_header_style) for c in cols])
                else:
                    table_data.append([Paragraph(c, table_text_style) for c in cols])
                continue
            elif in_table:
                # Table ended, build reportlab table
                in_table = False
                if table_data:
                    col_widths = [150] + [100] * (len(table_data[0]) - 1)
                    t = Table(table_data, colWidths=col_widths)
                    t.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#302b63")),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 12))
                    table_data = []
            
            # Title Heading
            if line_str.startswith("# "):
                title_text = line_str.replace("# ", "")
                story.append(Paragraph(title_text, title_style))
                story.append(Spacer(1, 10))
            # H2 Heading
            elif line_str.startswith("## "):
                h2_text = line_str.replace("## ", "")
                story.append(Paragraph(h2_text, h2_style))
                story.append(Spacer(1, 6))
            # Bullet point
            elif line_str.startswith("- ") or line_str.startswith("* "):
                bullet_text = line_str[2:]
                story.append(Paragraph(f"• {bullet_text}", body_style))
            # Numeric point
            elif re.match(r"^\d+\.", line_str):
                story.append(Paragraph(line_str, body_style))
            # Blockquote
            elif line_str.startswith(">"):
                quote_text = line_str.replace(">", "").strip()
                story.append(Paragraph(f"<i>{quote_text}</i>", body_style))
            # Empty line
            elif not line_str:
                story.append(Spacer(1, 6))
            # Plain paragraph
            else:
                # Simple markdown parsing for bold/italics
                parsed_str = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line_str)
                parsed_str = re.sub(r"\*(.*?)\*", r"<i>\1</i>", parsed_str)
                story.append(Paragraph(parsed_str, body_style))
                
        # Build Document
        doc.build(story)
        print(f"Success: Successfully generated PDF: {pdf_path}")
        
    except ImportError:
        print("ReportLab is not installed. Generating simplified PDF placeholder files.")
        print("Please install ReportLab: pip install reportlab")
        
        # Write PDF placeholder to notify user
        with open(pdf_path, "w", encoding="utf-8") as f:
            f.write(f"Placeholder for {os.path.basename(pdf_path)}\n")
            f.write(f"Please read the full Markdown report at: {os.path.abspath(md_path)}\n")
            f.write("To compile this markdown file into a PDF, install reportlab:\n")
            f.write("pip install reportlab && python scripts/generate_pdfs.py\n")


def main():
    # Resolve paths
    tech_md = os.path.join(REPORTS_DIR, "TECHNICAL_REPORT.md")
    tech_pdf = os.path.join(REPORTS_DIR, "TECHNICAL_REPORT.pdf")
    
    hall_md = os.path.join(REPORTS_DIR, "HALLUCINATION_ANALYSIS.md")
    hall_pdf = os.path.join(REPORTS_DIR, "HALLUCINATION_ANALYSIS.pdf")
    
    generate_pdf_from_md(tech_md, tech_pdf)
    generate_pdf_from_md(hall_md, hall_pdf)


if __name__ == "__main__":
    main()
