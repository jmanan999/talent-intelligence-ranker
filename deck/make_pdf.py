"""
deck/make_pdf.py — Generate deck.pdf from approach.md using ReportLab.

Converts the Marp markdown slides into a clean PDF with proper slide formatting.
Run: python deck/make_pdf.py
"""

import re
import textwrap
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Preformatted
)
from reportlab.pdfgen import canvas


PAGE = landscape(letter)
W, H = PAGE

PRIMARY   = colors.HexColor("#0f3460")
ACCENT    = colors.HexColor("#e94560")
LIGHT_BG  = colors.HexColor("#f8f9fa")
CODE_BG   = colors.HexColor("#1a1a2e")
CODE_FG   = colors.white
BODY      = colors.HexColor("#1a1a2e")


def slide_number_canvas(c, doc):
    """Draw page number and bottom bar."""
    c.saveState()
    c.setFillColor(ACCENT)
    c.rect(0, 0, W, 4, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#aaaaaa"))
    c.setFont("Helvetica", 9)
    c.drawRightString(W - 0.3*inch, 0.08*inch, f"Page {doc.page}")
    c.restoreState()


class SlidePDF:
    def __init__(self, output_path: str):
        self.output = output_path
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        self.story = []

    def _setup_styles(self):
        self.h1 = ParagraphStyle(
            "h1", parent=self.styles["Heading1"],
            fontSize=28, textColor=PRIMARY, spaceAfter=12,
            fontName="Helvetica-Bold",
        )
        self.h2 = ParagraphStyle(
            "h2", parent=self.styles["Heading2"],
            fontSize=18, textColor=PRIMARY, spaceAfter=8,
            fontName="Helvetica-Bold",
        )
        self.body = ParagraphStyle(
            "body", parent=self.styles["Normal"],
            fontSize=11, textColor=BODY, spaceAfter=6,
            fontName="Helvetica", leading=16,
        )
        self.code = ParagraphStyle(
            "code", parent=self.styles["Code"],
            fontSize=9, textColor=CODE_FG, backColor=CODE_BG,
            spaceAfter=8, fontName="Courier",
            leftIndent=12, rightIndent=12,
            borderPad=8, leading=12,
        )
        self.bullet = ParagraphStyle(
            "bullet", parent=self.body,
            leftIndent=20, bulletIndent=10,
            spaceAfter=4,
        )
        self.caption = ParagraphStyle(
            "caption", parent=self.body,
            fontSize=9, textColor=colors.HexColor("#666666"),
            fontName="Helvetica-Oblique",
        )

    def _parse_slides(self, md_text: str) -> list[str]:
        """Split by --- separators, skip frontmatter."""
        parts = re.split(r"^\s*---\s*$", md_text, flags=re.MULTILINE)
        slides = []
        for i, part in enumerate(parts):
            if i == 0 and part.strip().startswith("marp:"):
                continue  # skip frontmatter
            part = part.strip()
            if part:
                slides.append(part)
        return slides

    def _render_table(self, lines: list[str]) -> Table:
        """Render a markdown table."""
        rows = []
        for line in lines:
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows.append(cells)
        if len(rows) >= 2:
            rows.pop(1)  # remove separator row

        data = []
        for i, row in enumerate(rows):
            data.append([Paragraph(c, self.h2 if i == 0 else self.body) for c in row])

        col_w = (W - 2*inch) / max(len(data[0]), 1)
        t = Table(data, colWidths=[col_w] * len(data[0]))
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  PRIMARY),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
            ("GRID",        (0, 0), (-1, -1),  0.5, colors.HexColor("#dddddd")),
            ("PADDING",     (0, 0), (-1, -1),  6),
            ("VALIGN",      (0, 0), (-1, -1),  "TOP"),
        ]))
        return t

    def _slide_to_elements(self, slide_text: str) -> list:
        elems = []
        lines = slide_text.split("\n")
        in_code = False
        code_buf = []
        table_buf = []
        in_table = False

        for line in lines:
            # Code fence
            if line.startswith("```"):
                if not in_code:
                    in_code = True
                    code_buf = []
                else:
                    in_code = False
                    code_text = "\n".join(code_buf)
                    elems.append(Preformatted(
                        code_text, self.code,
                        maxLineLength=90,
                    ))
                continue
            if in_code:
                code_buf.append(line)
                continue

            # Tables
            if line.startswith("|"):
                in_table = True
                table_buf.append(line)
                continue
            if in_table and not line.startswith("|"):
                if table_buf:
                    elems.append(self._render_table(table_buf))
                    elems.append(Spacer(1, 6))
                table_buf = []
                in_table = False

            # Headers
            if line.startswith("# "):
                text = line[2:].strip()
                elems.append(Paragraph(text, self.h1))
                elems.append(HRFlowable(width="100%", thickness=2, color=ACCENT, spaceAfter=8))
                continue
            if line.startswith("## "):
                text = line[3:].strip()
                elems.append(Paragraph(text, self.h2))
                continue
            if line.startswith("### "):
                text = line[4:].strip()
                elems.append(Paragraph(f"<b>{text}</b>", self.body))
                continue

            # Bullets
            if line.startswith("- ") or line.startswith("* "):
                text = line[2:].strip()
                text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
                text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
                elems.append(Paragraph(f"• {text}", self.bullet))
                continue

            # Blockquote
            if line.startswith("> "):
                text = line[2:].strip()
                text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
                elems.append(Paragraph(f'<i>&#x201C;{text}&#x201D;</i>', self.caption))
                continue

            # Normal text
            text = line.strip()
            if text:
                text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
                text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
                text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
                elems.append(Paragraph(text, self.body))
            else:
                elems.append(Spacer(1, 4))

        if table_buf:
            elems.append(self._render_table(table_buf))

        return elems

    def build(self, md_path: str) -> None:
        text = Path(md_path).read_text(encoding="utf-8")
        slides = self._parse_slides(text)

        doc = SimpleDocTemplate(
            self.output,
            pagesize=PAGE,
            leftMargin=0.6*inch,
            rightMargin=0.6*inch,
            topMargin=0.5*inch,
            bottomMargin=0.35*inch,
        )

        for i, slide in enumerate(slides):
            elems = self._slide_to_elements(slide)
            self.story.extend(elems)
            if i < len(slides) - 1:
                self.story.append(PageBreak())

        doc.build(self.story, onFirstPage=slide_number_canvas, onLaterPages=slide_number_canvas)
        print(f"PDF generated: {self.output} ({Path(self.output).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    import sys
    md = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "approach.md")
    out = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent / "deck.pdf")
    SlidePDF(out).build(md)
