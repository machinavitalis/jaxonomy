# SPDX-License-Identifier: MIT
"""Preprocessing hook for `mkdocs-llmstxt`.

Drops the collapsed "Source code in ..." blocks that `mkdocstrings` renders
under every documented object. Those blocks reproduce the implementation
verbatim, and the signature plus docstring already appear directly above them —
so in a single-file corpus they are pure duplication. Removing them takes
`llms-full.txt` from roughly 3.5 MB to 1.4 MB, which is the difference between a
file an agent can pull in one request and one it cannot.

Only `details.mkdocstrings-source` is removed. The other `<details>` blocks on
these pages (input ports, output ports, notes, events) are documentation and are
left alone.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

# mkdocstrings-python wraps each source listing in this class.
_SOURCE_BLOCK_CLASS = "mkdocstrings-source"


def preprocess(soup: BeautifulSoup, output: str) -> None:
    """Strip duplicated source listings before Markdown conversion.

    Args:
        soup: Parsed HTML for one page, modified in place.
        output: Output path of the Markdown file being generated (unused; the
            source blocks only ever appear on mkdocstrings-rendered pages).
    """
    del output  # same treatment on every page

    for details in soup.find_all("details", class_=_SOURCE_BLOCK_CLASS):
        details.decompose()
