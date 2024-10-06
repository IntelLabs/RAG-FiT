# Generating RAG-FiT Documentation

## Installation

Install python packages required for building mkdocs documentation website.

``` sh
pip install -r docs/requirements.txt
```

## Adding new content

- Generate python docstrings using:

``` sh
cd <project root>
python docs/scripts/generate_docstring.py
```

- Add new API documentation to `mkdocs.yaml` in the appropriate section under the API main section.
- Add any new context as a markdown file and in the appropriate section under `nav` section in `mkdocs.yaml`.
- Check website functions correctly.

``` sh
mkdocs serve
```

Go to `http://127.0.0.1:8000/` for a live preview.

- Build and upload new website:

``` sh
mkdocs build
```