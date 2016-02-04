# Generate automatic html documentation using pdoc.
# pdoc is available here https://github.com/BurntSushi/pdoc
# and can be installed from Pypi: pip install pdoc

#python <<END
#import precession
#print "Generating documentation of precession, version", precession.__version__
#END

#pdoc --html --overwrite precession
#rm precession/*pyc precession/*/*pyc


##################

git checkout master

python <<END
import precession
print "Generating documentation of precession, version", precession.__version__
END

pdoc --html --overwrite precession

rm precession/*pyc precession/*/*pyc

temp1=`mktemp`
cp precession/index.html $temp1
temp2=`mktemp`
cp precession/test/index.html $temp2


git add *
git commit -m "generate_documentation.sh"
git push

git checkout gh-pages
mv $temp1 index.html
mv $temp2 test/index.html

git add *
git commit -m "generate_documentation.sh"
git push

rm -f $temp1 $temp2

git checkout master



