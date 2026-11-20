# XGA Documentation (`new-docs/`)

The new XGA documentation is being developed in parallel with the current Sphinx documentation (`docs/`), rather than replacing it in one fell swoop. The new documentation lives in this `new-docs/` top-level directory and is built using [MyST Markdown (`mystmd`)](https://mystmd.org/) / Jupyter Book 2, which provides a modern and powerful infrastructure for scientific documentation.

## Why MyST Markdown?

* **Rich UI & Formatting**: `mystmd` includes clean interactive GUI elements in the `book-theme` (dropdowns, tab sets, cards, grids, and admonitions).
* **Authorship, Citations, & Cross-Referencing**: First-class support for contributor metadata (`myst.yml`), BibTeX citations (`references.bib`), and hover-preview cross-referencing (for instance, hovering over a `{term}` reference to `glossary.md` or an internal link displays a pop-up preview window).
* **Executable Documentation (Text-Based Notebooks)**: `mystmd` is designed around executable documentation. Tutorials and case studies can be written as **MyST Markdown notebooks (`.md` via `jupytext`) rather than binary `.ipynb` files**, keeping git diffs clean and readable. Once our GitHub Actions / ReadTheDocs execution pipeline is wired up, notebook-based pages can be automatically re-executed when XGA changes.

## Building & Previewing Locally

### 1. Install Dependencies

Install `mystmd` and `jupytext` (either directly or via the `new-docs` optional dependency group in `pyproject.toml`):

```bash
pip install -e ".[new-docs]"
# or via conda / pip directly:
pip install mystmd jupytext
```

### 2. Start the Live Development Server

In your terminal, navigate to the `new-docs/` directory and run `myst start`:

```bash
cd new-docs
myst start
```

This starts a local web server (typically at `http://localhost:3000`) and prints the URL in your terminal. **In this mode, the website automatically rebuilds and refreshes whenever you save changes to the source files**, which is very useful when writing or editing pages.

### 3. Run a Strict Static HTML Build

To build the static HTML site (into `new-docs/_build/html`) and verify that all cross-references, citations, and links resolve without warnings:

```bash
cd new-docs
myst build --html --strict
```

## Drafting Color-Coding Convention (`_static/custom.css`)

While we are migrating and rewriting sections from the legacy `docs/` directory, we use CSS classes defined in `_static/custom.css` to visually distinguish where text came from so every new or modified passage can be reviewed by a human:

* **Default theme text color (uncolored)**: Verbatim text copied directly from the old Sphinx documentation (`docs/source/`).
* **Purple (`#9333ea`, `.edited-text`)**: Edits or updates made to text that was copied from the old documentation.
  * Inline usage: `<span class="edited-text">edited text</span>`
* **Red (`#dc2626`, `.placeholder-text`)**: Newly written / AI-generated placeholder text that requires human review and editing.
  * Inline usage: `<span class="placeholder-text">placeholder text</span>`
  * Block usage:
    ```markdown
    :::{div}
    :class: placeholder-text
    Block of placeholder content...
    :::
    ```
  * When wrapping nested directives (such as `::::{tab-set}` or `::::{dropdown}`), ensure the outer `:::::{div}` fence uses strictly more colons than any directive inside it.
  * To color only the GUI header of a mixed-content `{tab-set}`, `{dropdown}`, or admonition (`{warning}`, `{note}`, `{tip}`), wrap the directive in `:::{div}` with `:class: placeholder-tabs`, `:class: placeholder-dropdown`, or `:class: placeholder-admonition`.
