# Third-party licenses

This project's own code is licensed under MIT (see [LICENSE](LICENSE)).

It depends on the following third-party package under a different,
copyleft license:

- **[trafilatura](https://github.com/adbar/trafilatura)** — GPL-3.0.
  Used in `core/html.py` for HTML content extraction.

trafilatura is used as an ordinary PyPI runtime dependency (installed at
`pip install` time), not bundled or statically linked into this
repository. If you redistribute this project in a form that bundles or
statically links trafilatura, GPL-3.0's copyleft terms may apply to that
distribution. See trafilatura's own repository for its full license text.
