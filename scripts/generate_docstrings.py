# Ported from https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src = Path(__file__).parent.parent.parent / "ragfit"
excluded = ["__init__.py", "__pycache__"]

for path in sorted(src.rglob("*.py")):
    if any(path.name.__contains__(exclude) for exclude in excluded):
        print(f"Skipping {path}")
        continue
    print(f"Processing file: {path}")
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)
    parts = tuple(module_path.parts)
    print(f"{module_path} -> {doc_path} -> {full_doc_path} ; {module_path.parts}")

    # if parts[-1] == "__init__":
    #     parts = parts[:-1]
    #     doc_path = doc_path.with_name("index.md")
    #     full_doc_path = full_doc_path.with_name("index.md")
    # elif parts[-1] == "__main__":
    #     continue

    nav[parts] = doc_path.as_posix()
    print(f"{doc_path.as_posix()}")

    print(f"Writing to {full_doc_path}")
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: ragfit.{ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

    print("---\n")
# with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
#     nav_file.writelines(nav.build_literate_nav())
