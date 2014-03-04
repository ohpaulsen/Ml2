all: normal small
cclean: clean rmpdf

normal:
	pdflatex template.tex
	bibtex template
	pdflatex template.tex
	pdflatex template.tex
	evince template.pdf &

small:
	if [ -r ./template.pdf ]; \
	then \
		gs -sDEVICE=pdfwrite \
		   -dCompatibilityLevel=1.4 \
		   -dPDFSETTINGS=/ebook \
		   -dNOPAUSE \
		   -dQUIET \
		   -dBATCH \
		   -sOutputFile=template_ebook.pdf \
		   template.pdf; \
	else \
		echo "./template.pdf does not exist or is not readable."; \
	fi

clean:
	rm -vf template.aux \
		template.bbl \
		template.blg \
		template.log \
		template.toc \
		template.out

rmpdf:
	rm -vf template.pdf \
		template_ebook.pdf
