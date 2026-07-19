"""Regression tests for the pyGWRx source-documentation convention."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "pygwrx"
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
NUMPY_SECTION_PATTERN = re.compile(
    r"(?m)^\s*(Parameters|Returns|Raises|Attributes|Examples|Notes|"
    r"References|See Also|Yields)\s*\n\s*-{3,}\s*$"
)


def _iter_docstrings(tree: ast.AST):
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if body and isinstance(body, list) and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                yield value.value


def test_all_source_modules_use_the_standard_author_header():
    files = sorted(SOURCE_ROOT.rglob("*.py"))
    assert files

    for path in files:
        source = path.read_text(encoding="utf-8")
        assert source.startswith(
            "# SPDX-FileCopyrightText: 2026 Jinghao Hu\n"
            "# SPDX-License-Identifier: MIT\n\n"
            '"""'
        ), path
        assert '__author__ = "Jinghao Hu"' in source, path
        assert '__license__ = "MIT"' in source, path

        tree = ast.parse(source)
        module_docstring = ast.get_docstring(tree)
        assert module_docstring is not None, path
        assert "Author:\n    Jinghao Hu" in module_docstring, path


def test_docstrings_use_google_sections_and_english_text():
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for docstring in _iter_docstrings(tree):
            assert not NUMPY_SECTION_PATTERN.search(docstring), path
            assert not CJK_PATTERN.search(docstring), path


def test_source_comments_are_written_in_english():
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                assert not CJK_PATTERN.search(token.string), (
                    path,
                    token.start[0],
                    token.string,
                )


def test_mkdocs_uses_google_docstring_parsing():
    mkdocs_config = (PROJECT_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "docstring_style: google" in mkdocs_config
    assert "docstring_style: numpy" not in mkdocs_config
