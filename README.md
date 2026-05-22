# book_extractor

API REST que recibe fotos de páginas de libros, detecta fragmentos marcados manualmente con `[]` en el margen externo y extrae el texto con OCR.

## Requisitos

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) instalado
- Tesseract OCR instalado en el sistema (si `OCR_PROVIDER=tesseract`)

## Setup

```bash
uv sync --extra dev
cp .env.example .env
```

## Correr la API

```bash
uv run book-extractor
# o equivalente:
uv run uvicorn app.main:app --reload
```

### Solución errores import open cv 

```bash
uv sync --extra dev
```
Ctrl+Shift+P → "Python: Select Interpreter" → elegir .venv\Scripts\python.exe.
Ctrl+Shift+P → "Developer: Reload Window"

La API queda en `http://localhost:8000` y expone `POST /extract` y `GET /health`.

## Tests

```bash
uv run pytest
```

## Variables de entorno

Ver `.env.example`. Provider OCR configurable: `tesseract` o `google_vision`.

## Desplegar tesseract
https://github.com/naptha/tesseract.js#tesseractjs

https://github.com/tesseract-ocr/tesseract
