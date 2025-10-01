# Makefile for LaTeX paper
PAPER=paper

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(PAPER).bib
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc

.PHONY: all clean