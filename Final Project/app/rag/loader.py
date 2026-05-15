from pathlib import Path

from markitdown import MarkItDown

md_converter = MarkItDown()


def load_pdf(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    result = md_converter.convert(str(path))
    text = result.text_content
    if not text or not text.strip():
        raise ValueError(
            f"No se pudo extraer texto de {path}. "
            "Verifica que el PDF no sea escaneado."
        )
    return text
