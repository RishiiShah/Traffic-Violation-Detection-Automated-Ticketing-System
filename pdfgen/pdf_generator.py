import pdfkit
from jinja2 import Template
import datetime
from pathlib import Path

def generate_pdf(data: dict,
                 template_path: str,
                 wkhtmltopdf_path: str = None) -> Path:
    """
    Renders `template_path` with `data` and emits a PDF, returning its Path.
    """
    # 1) Load & render Jinja2 template
    with open(template_path, 'r', encoding='utf-8') as f:
        tmpl = Template(f.read())
    html = tmpl.render(data)

    # 2) Build output filename: challan_<no>_<YYYYMMDD>.pdf
    no   = data.get('challan_no', 'challan')
    date = datetime.datetime.now().strftime("%Y%m%d")
    out  = Path(f"challan_{no}_{date}.pdf").resolve()

    # 3) Configure wkhtmltopdf if given
    config = None
    if wkhtmltopdf_path:
        config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)

    # 4) Generate PDF
    pdfkit.from_string(
        html,
        str(out),
        options={"enable-local-file-access": ""},
        configuration=config
    )

    return out
