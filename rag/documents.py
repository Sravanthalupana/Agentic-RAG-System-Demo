import hashlib
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

import tiktoken
from docx import Document
from docx.table import Table
from pypdf import PdfReader

from rag.models import Chunk, Section

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARS = 2_000_000


def load_document(filename: str, data: bytes) -> list[Section]:
    source = PurePosixPath(filename.replace("\\", "/")).name
    suffix = PurePosixPath(source).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md"}:
        raise ValueError("Supported formats: PDF, DOCX, TXT, MD.")
    if not data or len(data) > MAX_FILE_BYTES:
        raise ValueError("Document must be nonempty and at most 20 MB.")
    sections: list[Section] = []
    try:
        if suffix in {".txt", ".md"}:
            sections = [Section(source, "text", data.decode("utf-8-sig"))]
        elif suffix == ".pdf":
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted or len(reader.pages) > 500:
                raise ValueError("PDF must be unencrypted and at most 500 pages.")
            sections = [
                Section(source, f"page {i}", page.extract_text() or "")
                for i, page in enumerate(reader.pages, 1)
            ]
        else:
            with ZipFile(BytesIO(data)) as archive:
                if sum(info.file_size for info in archive.infolist()) > 50 * 1024 * 1024:
                    raise ValueError("DOCX uncompressed content exceeds 50 MB.")
            document = Document(BytesIO(data))
            for i, block in enumerate(document.iter_inner_content(), 1):
                text = (
                    "\n".join(" | ".join(cell.text for cell in row.cells) for row in block.rows)
                    if isinstance(block, Table)
                    else block.text
                )
                sections.append(Section(source, f"block {i}", text))
    except (UnicodeError, BadZipFile) as exc:
        raise ValueError("Unreadable document; text files must use UTF-8.") from exc
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("Could not read this document; check the file format.") from exc
    sections = [section for section in sections if section.text.strip()]
    if not sections:
        raise ValueError("No readable text found. Scanned PDFs require OCR before upload.")
    if sum(len(section.text) for section in sections) > MAX_TEXT_CHARS:
        raise ValueError("Extracted text exceeds 2 million characters.")
    return sections


def chunk_sections(
    sections: list[Section], chunk_tokens: int = 500, overlap_tokens: int = 75
) -> list[Chunk]:
    if chunk_tokens < 4 or not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("Use chunk_tokens >= 4 and 0 <= overlap_tokens < chunk_tokens.")
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks: list[Chunk] = []
    for section in sections:
        fingerprint = hashlib.sha256(section.text.encode()).hexdigest()
        tokens = encoding.encode(section.text, disallowed_special=())
        start = 0
        while start < len(tokens):
            end = min(start + chunk_tokens, len(tokens))
            while True:
                try:
                    text = encoding.decode(tokens[start:end], errors="strict")
                    break
                except UnicodeDecodeError:
                    end -= 1
                    if end <= start:
                        raise ValueError("Chunk token budget is too small for this text.") from None
            if text.strip():
                identity = f"{section.source}\0{section.location}\0{start}\0{fingerprint}"
                chunks.append(
                    Chunk(
                        hashlib.sha256(identity.encode()).hexdigest()[:20],
                        section.source,
                        section.location,
                        text,
                    )
                )
            if end == len(tokens):
                break
            start = max(start + 1, end - overlap_tokens)
            while start < end:
                try:
                    encoding.decode(tokens[start:end], errors="strict")
                    break
                except UnicodeDecodeError:
                    start += 1
    return chunks
