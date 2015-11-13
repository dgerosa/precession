# Generate automatic html documentation using pdoc.
# pdoc is available here https://github.com/BurntSushi/pdoc
# and can be installed from Pypi: pip install pdoc

pdoc --html --html-dir=documentation precession
mv documentation/precession/* documentation
rm -r documentation/precession