PDF     = poster.pdf
SRC     = poster.tex
PREVIEW = preview-1.png

.PHONY: all preview clean open

all: $(PDF)

$(PDF): $(SRC) $(wildcard figures/*.png)
	pdflatex -interaction=nonstopmode $(SRC)

# low-res PNG for checking the layout on screen
preview: $(PDF)
	pdftoppm -png -r 40 $(PDF) preview

# high-res PNG, e.g. for a slide or a Slack post (~300 MB at 150 dpi — careful)
png: $(PDF)
	pdftoppm -png -r 150 $(PDF) poster-hires

open: $(PDF)
	open $(PDF)

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb $(PREVIEW)

distclean: clean
	rm -f $(PDF) poster-hires*.png
