.PHONY: all figures clean view distclean

MAIN := presentation
LATEX := xelatex
LATEXFLAGS := -interaction=nonstopmode -shell-escape

# Default target: always rebuild the PDF
all:
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# Generate matplotlib figures (run manually when needed)
figures:
	python3 make_figures.py

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
	rm -f figures/*.pdf figures/*.png
