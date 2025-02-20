from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages


def savefig(fig, name, formats=["pdf", "png", "svg"], path="figures"):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    for fmt in formats:
        if fmt == 'pdf':
            with PdfPages(path / f"{name}.pdf") as pdf:
                pdf.savefig(fig, bbox_inches="tight")
        elif fmt == 'png':
            fig.savefig(path / f"{name}.png", bbox_inches="tight", dpi=300)
        elif fmt == 'svg':
            fig.savefig(path / f"{name}.svg", bbox_inches="tight")
        else:
            raise ValueError(f'Unsupported format: {fmt}.')
