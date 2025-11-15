#!/bin/bash
# Compile LaTeX report to PDF

echo "Compiling report.tex to PDF..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found. Please install LaTeX distribution:"
    echo "  macOS: brew install --cask mactex-no-gui"
    echo "  or: brew install basictex"
    exit 1
fi

# Compile twice for references
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex

# Clean up auxiliary files
rm -f report.aux report.log report.out

echo "Done! PDF generated: report.pdf"
open report.pdf 2>/dev/null || echo "Please open report.pdf manually"
