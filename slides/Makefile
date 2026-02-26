.PHONY: all clean view

MAIN := presentation
LATEX := xelatex
LATEXFLAGS := -interaction=nonstopmode -shell-escape

# Default target
all: $(MAIN).pdf

# Build PDF using xelatex (better font rendering)
$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# View the PDF
view: $(MAIN).pdf
	open $(MAIN).pdf

# Clean temporary files
clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).nav $(MAIN).out $(MAIN).snm $(MAIN).toc
	rm -f $(MAIN).vrb $(MAIN).blg $(MAIN).bbl
	rm -rf _minted-* .minted*
	rm -f *.synctex.gz

# Clean everything including PDF
distclean: clean
	rm -f $(MAIN).pdf
