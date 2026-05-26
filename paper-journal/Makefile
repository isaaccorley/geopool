TEX = main
PDF = $(TEX).pdf

.PHONY: build clean watch

build: $(PDF)

TABLES = $(wildcard tables/*.tex)

$(PDF): $(TEX).tex refs.bib IEEEtran.cls IEEEtran.bst $(TABLES) figures/nd_rule.pdf
	pdflatex $(TEX)
	bibtex $(TEX)
	pdflatex $(TEX)
	pdflatex $(TEX)

figures/nd_rule.pdf: scripts/gen_nd_rule.py
	python3 scripts/gen_nd_rule.py

clean:
	rm -f $(TEX).{aux,bbl,blg,log,out,fdb_latexmk,fls,synctex.gz,pdf}

watch:
	latexmk -pvc -pdf $(TEX)
