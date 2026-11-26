# Third-party licenses

This project's own code is licensed under MIT (see [LICENSE](LICENSE)).

It depends on the following third-party packages under different,
copyleft licenses:

- **[trafilatura](https://github.com/adbar/trafilatura)** — GPL-3.0.
  Used in `core/html.py` for HTML content extraction.

trafilatura is used as an ordinary PyPI runtime dependency (installed at
`pip install` time), not bundled or statically linked into this
repository. If you redistribute this project in a form that bundles or
statically links trafilatura, GPL-3.0's copyleft terms may apply to that
distribution. See trafilatura's own repository for its full license text.

- **[PyMuPDF](https://github.com/pymupdf/PyMuPDF)** (`pymupdf`) — dual
  licensed: GNU AGPL-3.0 or an Artifex commercial license. Used in the PDF
  text extraction module (`pdf_text.py`) for the `pdf` command.

PyMuPDF is likewise an ordinary PyPI runtime dependency, not bundled in this
repository. AGPL-3.0 also covers use over a network: if you offer a modified
or combined version of this project as a network service, or redistribute it
with PyMuPDF bundled, the AGPL-3.0 terms (source availability for that
combined work) may apply unless you hold a commercial license from Artifex.
See PyMuPDF's own repository for its full license text.
