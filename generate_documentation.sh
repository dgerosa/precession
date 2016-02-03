# Generate automatic html documentation using pdoc.
# pdoc is available here https://github.com/BurntSushi/pdoc
# and can be installed from Pypi: pip install pdoc

python <<END
import precession
print "Generating documentation of precession, version", precession.__version__
END

pdoc --html --overwrite precession
