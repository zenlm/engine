.PHONY: all clean

all: paper.pdf

paper.pdf: paper.tex references.bib
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf
