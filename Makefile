PY=python3
LINT=pylint
DOCS=$(basename $(wildcard *.rst)) $(basename $(wildcard tut/*.rst))

all: install html

build: setup.py src/*
	$(PY) setup.py sdist

install: src/* setup.py 
	$(PY) setup.py install

test:	src/*.py test/*
	for t in test/*; do $(PY) $$t; done;

checkstyle: src/*.py tut/*.py
	$(LINT) src
	for t in test/*; do $(LINT) $$t; done;

html: $(addsuffix .html, $(DOCS))

setup.py: version raw.setup.py
	cat raw.setup.py | sed s/VERSION/$(shell cat version)/g > setup.py
	chmod +x setup.py

pdf: $(addsuffix .pdf, $(DOCS))

%.pdf: %.rst
	rst2pdf $<

%.html: %.rst
	rst2html $< > %.html

clean:
	rm -rf dist/ build/ *.egg-info/ setup.py *.pdf
