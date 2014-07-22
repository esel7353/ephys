PY=python3
LINT=pylint

all: install test checkstyle

install: src/* setup.py README.md
	sudo $(PY) setup.py

test:	src/*.py test/*
	for t in test/*; do $(PY) $$t; done;

checkstyle: src/*.py tut/*.py
	$(LINT) src
	for t in test/*; do $(LINT) $$t; done;
