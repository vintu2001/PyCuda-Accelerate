project = "PyCuda-Accelerate"
author = "Your Name"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "breathe",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

breathe_projects = {"PyCudaAccelerate": "../build/doxygen/xml"}
breathe_default_project = "PyCudaAccelerate"

html_theme = "sphinx_rtd_theme"
