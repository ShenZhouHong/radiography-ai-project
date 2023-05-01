# set the shell to use bash
SHELL := /bin/bash

# define the build command
BUILD_COMMAND := SOURCE_DATE_EPOCH=$(shell git log -1 --pretty=%ct) latexmk -pdf -lualatex -shell-escape

# find all .tex files in the directory, excluding preamble.tex and standalone-template.tex
TEX_FILES := $(filter-out preamble.tex standalone-template.tex,$(wildcard *.tex))

# define the default target to build all LaTeX files
all: $(TEX_FILES:.tex=.pdf)

# define a rule to build a single PDF from a single .tex file
%.pdf: %.tex
	$(BUILD_COMMAND) $<

# define a rule to clean up generated files
clean:
	latexmk -c
	rm -f *.pdf
