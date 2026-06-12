"""
Compile project Markdown reports into polished PDF files.

The parser supports the report features used in this repository: headings,
paragraphs, bullets, tables, fenced code blocks, page breaks, and local images.
"""

import os
import re
import sys
from html import escape

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import REPORTS_DIR


def _md_inline(text: str) -> str:
    """Convert a small subset of Markdown inline syntax into ReportLab HTML."""
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    return text


def _flush_table(story, table_data, col_count):
    """Append a styled ReportLab table to the story."""
    if not table_data:
        return

    from reportlab.lib import colors
    from reportlab.platypus import Spacer, Table, TableStyle

    available_width = 504
    col_widths = [available_width / max(col_count, 1)] * max(col_count, 1)
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D47A1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D6DEE8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))


def _add_image(story, image_path, caption, md_dir):
    """Append a local image with bounded PDF dimensions."""
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Image as RLImage, Paragraph, Spacer

    resolved = image_path
    if not os.path.isabs(resolved):
        resolved = os.path.join(md_dir, resolved)
    resolved = os.path.normpath(resolved)

    if not os.path.exists(resolved):
        story.append(Paragraph(
            f"<i>Image not found: {_md_inline(image_path)}</i>",
            getSampleStyleSheet()["BodyText"],
        ))
        return

    with PILImage.open(resolved) as img:
        width, height = img.size

    max_width = 504
    max_height = 270
    scale = min(max_width / width, max_height / height, 1.0)
    story.append(RLImage(resolved, width=width * scale, height=height * scale))

    if caption:
        caption_style = ParagraphStyle(
            name="ImageCaption",
            parent=getSampleStyleSheet()["BodyText"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#5B6673"),
            alignment=1,
            spaceAfter=10,
        )
        story.append(Paragraph(_md_inline(caption), caption_style))
    story.append(Spacer(1, 8))


def _page_footer(canvas, doc):
    """Draw a small page footer and page number."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColorRGB(0.34, 0.39, 0.45)
    canvas.drawString(42, 24, "ClinicGuard-ReportGen | Research prototype, not for clinical diagnosis")
    canvas.drawRightString(570, 24, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf_from_md(md_path: str, pdf_path: str):
    """Compile a Markdown file into a PDF file using ReportLab."""
    print(f"Compiling {md_path} into {pdf_path}...")

    if not os.path.exists(md_path):
        print(f"Error: source file {md_path} does not exist.")
        return

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer

        with open(md_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=42,
            leftMargin=42,
            topMargin=48,
            bottomMargin=42,
            title=os.path.splitext(os.path.basename(pdf_path))[0],
        )
        story = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            name="TitleStyle",
            parent=styles["Heading1"],
            fontSize=23,
            leading=28,
            textColor=colors.HexColor("#0D47A1"),
            spaceAfter=12,
        )
        h2_style = ParagraphStyle(
            name="H2Style",
            parent=styles["Heading2"],
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#0D47A1"),
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
        )
        h3_style = ParagraphStyle(
            name="H3Style",
            parent=styles["Heading3"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#263238"),
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        )
        body_style = ParagraphStyle(
            name="BodyStyle",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#263238"),
            spaceAfter=7,
        )
        bullet_style = ParagraphStyle(
            name="BulletStyle",
            parent=body_style,
            leftIndent=14,
            firstLineIndent=-8,
        )
        code_style = ParagraphStyle(
            name="CodeStyle",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=7.6,
            leading=9.5,
            textColor=colors.HexColor("#263238"),
            backColor=colors.HexColor("#F5F7FA"),
            borderColor=colors.HexColor("#D6DEE8"),
            borderWidth=0.4,
            borderPadding=6,
            spaceAfter=9,
        )
        table_text_style = ParagraphStyle(
            name="TableTextStyle",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#263238"),
        )
        table_header_style = ParagraphStyle(
            name="TableHeaderStyle",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            textColor=colors.white,
            fontName="Helvetica-Bold",
        )

        in_table = False
        table_data = []
        in_code = False
        code_lines = []
        md_dir = os.path.dirname(os.path.abspath(md_path))

        for line in lines:
            line_str = line.strip()

            if line_str.startswith("```"):
                if in_code:
                    story.append(Preformatted("\n".join(code_lines), code_style))
                    code_lines = []
                    in_code = False
                else:
                    in_code = True
                continue

            if in_code:
                code_lines.append(line.rstrip("\n"))
                continue

            if line_str.startswith("|"):
                cols = [c.strip() for c in line_str.split("|")[1:-1]]
                if not cols or all(not c for c in cols):
                    continue
                in_table = True
                if all(re.match(r"^:?-{2,}:?$", c) for c in cols):
                    continue
                if not table_data:
                    table_data.append([Paragraph(_md_inline(c), table_header_style) for c in cols])
                else:
                    table_data.append([Paragraph(_md_inline(c), table_text_style) for c in cols])
                continue
            elif in_table:
                in_table = False
                _flush_table(story, table_data, len(table_data[0]) if table_data else 0)
                table_data = []

            image_match = re.match(r"!\[(.*?)\]\((.*?)\)", line_str)
            if image_match:
                _add_image(story, image_match.group(2), image_match.group(1), md_dir)
                continue

            if line_str == '<div class="page-break"></div>':
                story.append(PageBreak())
                continue

            if line_str.startswith("# "):
                story.append(Paragraph(_md_inline(line_str.replace("# ", "", 1)), title_style))
                story.append(Spacer(1, 10))
            elif line_str.startswith("## "):
                story.append(Paragraph(_md_inline(line_str.replace("## ", "", 1)), h2_style))
                story.append(Spacer(1, 6))
            elif line_str.startswith("### "):
                story.append(Paragraph(_md_inline(line_str.replace("### ", "", 1)), h3_style))
            elif line_str.startswith("- ") or line_str.startswith("* "):
                story.append(Paragraph(f"- {_md_inline(line_str[2:])}", bullet_style))
            elif re.match(r"^\d+\.", line_str):
                story.append(Paragraph(_md_inline(line_str), body_style))
            elif line_str.startswith(">"):
                story.append(Paragraph(f"<i>{_md_inline(line_str.replace('>', '').strip())}</i>", body_style))
            elif not line_str:
                story.append(Spacer(1, 6))
            else:
                story.append(Paragraph(_md_inline(line_str), body_style))

        if in_code and code_lines:
            story.append(Preformatted("\n".join(code_lines), code_style))

        if in_table and table_data:
            _flush_table(story, table_data, len(table_data[0]) if table_data else 0)

        doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
        print(f"Success: Successfully generated PDF: {pdf_path}")

    except ImportError:
        print("ReportLab is not installed. Generating simplified PDF placeholder files.")
        print("Please install ReportLab: pip install reportlab")

        with open(pdf_path, "w", encoding="utf-8") as f:
            f.write(f"Placeholder for {os.path.basename(pdf_path)}\n")
            f.write(f"Please read the full Markdown report at: {os.path.abspath(md_path)}\n")
            f.write("To compile this markdown file into a PDF, install reportlab:\n")
            f.write("pip install reportlab && python scripts/generate_pdfs.py\n")


def main():
    tech_md = os.path.join(REPORTS_DIR, "TECHNICAL_REPORT.md")
    tech_pdf = os.path.join(REPORTS_DIR, "TECHNICAL_REPORT.pdf")

    hall_md = os.path.join(REPORTS_DIR, "HALLUCINATION_ANALYSIS.md")
    hall_pdf = os.path.join(REPORTS_DIR, "HALLUCINATION_ANALYSIS.pdf")

    generate_pdf_from_md(tech_md, tech_pdf)
    generate_pdf_from_md(hall_md, hall_pdf)


if __name__ == "__main__":
    main()
