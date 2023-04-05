# We set the modification and creation date of the PDF to the latest git commit
# This way, PDF builds are reproducible across commits. If you are not using git,
# set this value to a constant like 0, or the current date using command date +%s
COMMIT_EPOCH = $(shell git log -1 --pretty=%ct)

# Multiple source names can be added by seperating entries with a space ' '
source_names = implementation proposal

# Makes sure latexmk always runs
.PHONY: all clean delete $(source_names:%=%.pdf)
all: $(source_names:%=%.pdf)

# MAIN LATEXMK RULE
$(source_names:%=%.pdf): $(source_names:%=%.tex)
	# We set the modification and creation date of the PDF to the latest git commit
	# This way, PDF builds are reproducible across commits.
	echo $(COMMIT_EPOCH)
	
	# This makefile uses lualatex by default. This way, it supports unicode
	# characters in the latex source files. If you wish to use latex instead of
	# lualatex, simply remove the -lualatex argument
	SOURCE_DATE_EPOCH=$(COMMIT_EPOCH) latexmk -pdf -lualatex -shell-escape -use-make $<
	sha256sum $@

	# For syntax highlighting with minted, enable -shell-escape:
	# latexmk -pdf -lualatex -shell-escape -use-make $<

clean:
	latexmk -c

delete:
	latexmk -C
