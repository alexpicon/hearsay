# Author: Alex Picon <alexnpc@me.com>
"""Stamp author credits into the repo's source files and web pages.

Safe to run repeatedly: a file that already carries the credit is left alone.
Data files (``data/``, JSON, CSV, PDF, lockfiles) are skipped because they
either hold third-party content or cannot take comments.
"""

import logging
import re
import subprocess
from pathlib import Path

NAME = "Alex Picon"
EMAIL = "alexnpc@me.com"
TAG = f"{NAME} <{EMAIL}>"
ROOT = Path(__file__).resolve().parent.parent
MARKER = 'data-credit="alex-picon"'
KEEP_FIRST = ("#!", "@charset", "<?xml", "<!doctype", "<!DOCTYPE")

HTML_META = f'<meta name="author" content="{NAME}, {EMAIL}">'
HTML_BADGE = (
    f"<div {MARKER} "
    'style="position:fixed;left:10px;bottom:8px;z-index:2147483647;'
    "font:500 11px/1.2 system-ui,sans-serif;padding:4px 8px;border-radius:6px;"
    'background:rgba(0,0,0,.45);color:#fff;opacity:.85">'
    f'Built by {NAME} · <a href="mailto:{EMAIL}" title="Email {NAME}" '
    f'style="color:inherit">{EMAIL}</a></div>'
)
MD_LINE = f'*Author: {NAME} ([{EMAIL}](mailto:{EMAIL} "Email {NAME}"))*'

COMMENT_STYLES: dict[str, str] = {
    ".css": f"/* Author: {TAG} */",
    ".js": f"// Author: {TAG}",
    ".mjs": f"// Author: {TAG}",
    ".py": f"# Author: {TAG}",
    ".sh": f"# Author: {TAG}",
    ".toml": f"# Author: {TAG}",
    ".svg": f"<!-- Author: {TAG} -->",
}

logger = logging.getLogger(__name__)


def candidate_files() -> list[Path]:
    """List tracked and untracked-but-not-ignored files worth crediting.

    Returns:
        Absolute paths of the files to stamp, excluding ``data/`` content.

    """
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    files: list[Path] = []
    for rel in sorted(set(listing)):
        if rel.startswith("data/") and rel != "data/README.md":
            continue
        path = ROOT / rel
        if path.is_file():
            files.append(path)
    return files


def prepend_line(text: str, line: str) -> str:
    """Put ``line`` at the top, after a shebang/charset/doctype if present.

    Args:
        text: The original file content.
        line: The credit line to insert.

    Returns:
        The content with the credit line inserted.

    """
    first, sep, rest = text.partition("\n")
    if first.startswith(KEEP_FIRST):
        return f"{first}\n{line}\n{rest}" if sep else f"{first}\n{line}\n"
    return f"{line}\n{text}"


def stamp_html(text: str) -> str:
    """Add an author meta tag and a visible credit badge to an HTML page.

    Args:
        text: The original HTML.

    Returns:
        The HTML with credits, or the input unchanged if already credited.

    """
    if MARKER in text:
        return text
    author_meta = re.compile(r'<meta\s+name=["\']author["\'][^>]*>', re.IGNORECASE)
    if author_meta.search(text):
        text = author_meta.sub(HTML_META, text, count=1)
    else:
        head = re.search(r"<head[^>]*>", text, re.IGNORECASE)
        charset = re.search(r"<meta\s+charset[^>]*>", text, re.IGNORECASE)
        anchor = head or charset
        if anchor:
            text = f"{text[: anchor.end()]}{HTML_META}{text[anchor.end() :]}"
        else:
            text = prepend_line(text, HTML_META)
    closing = [m.start() for m in re.finditer(r"</body>", text, re.IGNORECASE)]
    if closing:
        cut = closing[-1]
        return f"{text[:cut]}{HTML_BADGE}{text[cut:]}"
    return f"{text.rstrip()}\n{HTML_BADGE}\n"


def stamp_markdown(text: str) -> str:
    """Add an author line under the document title.

    Args:
        text: The original Markdown.

    Returns:
        The Markdown with the author line, or unchanged if already credited.

    """
    if EMAIL in text:
        return text
    first, sep, rest = text.partition("\n")
    if first.startswith("# "):
        return f"{first}\n\n{MD_LINE}\n\n{rest.lstrip(chr(10))}"
    return f"{MD_LINE}\n\n{text}"


def stamp(path: Path) -> bool:
    """Credit one file in place.

    Args:
        path: The file to stamp.

    Returns:
        True if the file was changed.

    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError, OSError:
        return False
    suffix = path.suffix.lower()
    if suffix == ".html":
        new = stamp_html(text)
    elif suffix == ".md":
        new = stamp_markdown(text)
    elif path.name == ".gitignore":
        new = text if EMAIL in text else f"# Author: {TAG}\n{text}"
    elif suffix in COMMENT_STYLES:
        new = text if EMAIL in text else prepend_line(text, COMMENT_STYLES[suffix])
    else:
        return False
    if new == text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def main() -> None:
    """Stamp every candidate file and log how many changed."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    changed = [p for p in candidate_files() if stamp(p)]
    for path in changed:
        logger.info("credited %s", path.relative_to(ROOT))
    logger.info("%d file(s) credited", len(changed))


if __name__ == "__main__":
    main()
