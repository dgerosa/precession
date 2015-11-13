# Generate automatic html documentation using pdoc.
# pdoc is available here https://github.com/BurntSushi/pdoc
# and can be installed from Pypi: pip install pdoc

python <<END
import precession
print "Generating documentation of precession, version", precession.__version__, "(from Pypi)."
END

pdoc --html --html-dir=temp precession
mv temp/precession/* .
rm -r temp