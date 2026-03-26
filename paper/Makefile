TEX = iclr2026_conference
PDF = $(TEX).pdf

.PHONY: build clean watch

build: $(PDF)

$(PDF): $(TEX).tex $(TEX).bib $(TEX).sty math_commands.tex knn_table.tex linear_table.tex
	pdflatex $(TEX)
	bibtex $(TEX)
	pdflatex $(TEX)
	pdflatex $(TEX)

clean:
	rm -f $(TEX).{aux,bbl,blg,log,out,fdb_latexmk,fls,synctex.gz,pdf}

watch:
	latexmk -pvc -pdf $(TEX)
