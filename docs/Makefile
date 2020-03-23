# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
# Zero static directory
STATICDIR = _static

### Dynamic plot dependencies
# Static files
ZEROSTATICDEPS = $(STATICDIR)/liso-input-node-graph.svg
# Native plots with LISO files.
ZEROLISODEPS = $(STATICDIR)/liso-two-noises.svg
# Native/LISO script comparisons.
ZEROLISOCOMPAREDEPS = $(STATICDIR)/liso-compare-response.svg
# CLI example plots.
CLIOPAMPGAINDEP = $(STATICDIR)/cli-opamp-gain.svg
ZEROCLIPLOTDEPS = $(CLIOPAMPGAINDEP)

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make a single large HTML file"
	@echo "  pickle     to make pickle files"
	@echo "  json       to make JSON files"
	@echo "  htmlhelp   to make HTML files and a HTML help project"
	@echo "  qthelp     to make HTML files and a qthelp project"
	@echo "  applehelp  to make an Apple Help Book"
	@echo "  devhelp    to make HTML files and a Devhelp project"
	@echo "  epub       to make an epub"
	@echo "  epub3      to make an epub3"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  latexpdfja to make LaTeX files and run them through platex/dvipdfmx"
	@echo "  text       to make text files"
	@echo "  man        to make manual pages"
	@echo "  texinfo    to make Texinfo files"
	@echo "  info       to make Texinfo files and run them through makeinfo"
	@echo "  gettext    to make PO message catalogs"
	@echo "  changes    to make an overview of all changed/added/deprecated items"
	@echo "  xml        to make Docutils-native XML files"
	@echo "  pseudoxml  to make pseudoxml-XML files for display purposes"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  doctest    to run all doctests embedded in the documentation (if enabled)"
	@echo "  coverage   to run coverage check of the documentation (if enabled)"
	@echo "  dummy      to check syntax errors of document sources"

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf api/*

.PHONY: apidoc
apidoc:
	sphinx-apidoc -o developers/api -e ../zero

.PHONY: static
static: plots cli-plots

.PHONY: plots
plots: $(ZEROSTATICDEPS) $(ZEROPYTHONDEPS) $(ZEROLISODEPS) $(ZEROLISOCOMPAREDEPS)

.PHONY: cli-plots
cli-plots: $(ZEROCLIPLOTDEPS)

# CLI op-amp gain plot.
$(CLIOPAMPGAINDEP):
	@echo "Generating $@"
	zero library search "gbw > 800M & ((vnoise < 10n & inoise < 10p) | (vnoise < 100n & inoise < 1p)) & model != OP00" --no-plot-gain --save-gain-figure $@ --fstop 1M

# Generate SVG plot of two noise example.
$(STATICDIR)/liso-two-noises.svg:
	@echo "Generating $@"
	zero liso $(STATICDIR)/liso-two-noises/noise1.fil $(STATICDIR)/liso-two-noises/noise2.fil --no-plot --save-figure $@

# Generate SVG plots comparing Zero to LISO using LISO files.
$(STATICDIR)/%.svg: $(STATICDIR)/liso-compare/%.fil
	@echo "Generating $@ from $<"
	zero liso $< --compare --no-plot --save-figure $@

.PHONY: html
html: apidoc static
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

.PHONY: latex
latex: apidoc static
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

.PHONY: latexpdf
latexpdf: apidoc static
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

.PHONY: man
man: apidoc static
	$(SPHINXBUILD) -b man $(ALLSPHINXOPTS) $(BUILDDIR)/man
	@echo
	@echo "Build finished. The manual pages are in $(BUILDDIR)/man."
