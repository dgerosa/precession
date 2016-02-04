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

# Be sure your working branch is clean
#if [[ $((git-status---porcelain)) -eq 0 ]]; then 
if [ "$(git status --porcelain)" ]; then 
  echo "Please, clean your working directory first."
  exit 1
else 
  echo "Generating documentation"; 
fi
# 
# # Check version of the code seen by pdoc
# python <<END
# import precession
# print "Python module precession, version", precession.__version__
# END
# 
# # Generate documentation using pdc
# pdoc --html --overwrite precession
# # Get rid of precompiled files
# rm precession/*pyc precession/*/*pyc
# 
# # Move html files somewhere else
# temp1=`mktemp`
# cp precession/index.html $temp1
# temp2=`mktemp`
# cp precession/test/index.html $temp2
# 
# # Commit new html to master branch
# git add *
# git commit -m "generate_documentation.sh"
# git push
# 
# # Move html files to gh-pages branch (directories there should exist)
# git checkout gh-pages
# mv $temp1 index.html
# mv $temp2 test/index.html
# 
# # Commit new html to gh-pages branch
# git add *
# git commit -m "generate_documentation.sh"
# git push
# 
# # Get rid of temp files
# rm -f $temp1 $temp2
# 
# git checkout master
# 
# 

