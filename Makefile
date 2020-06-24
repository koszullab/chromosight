.PHONY: build install test clean deploy apidoc

install:
	pip install -e .

uninstall:
	pip uninstall chromosight

clean:
	rm -rf build/ dist/

build: clean
	python setup.py sdist bdist_wheel

deploy: build
	twine upload dist/*

apidoc:
	sphinx-apidoc -f -o docs/api chromosight

test:
	nose2 -s tests/

