#Usage: bash generate_documentation.sh [-h|--help] [-web]

web=0
while test $# -gt 0; do
        case "$1" in
                -h|--help)
                        echo "Usage: bash generate_documentation.sh [options]"
                        echo " "
                        echo "Generate automatic html documentation using pdoc"
                        echo "pdoc is available here https://github.com/BurntSushi/pdoc"
                        echo "and can be installed from Pypi: pip install pdoc."
                        echo "If -web, documentation is pushed to a dedicated git branch (gh-pages)"
                        echo "and published at http://dgerosa.github.io/precession/"
                        echo " "
                        echo "options:"
                        echo "   -h, --help       show brief help"
                        echo "   -web             push documentation to http://dgerosa.github.io/precession/"
                        exit 0
                        ;;
                -web)
                        shift
                        web=1
                        shift
                        ;;
                *)
                        break
                        ;;
        esac
done


if [ $web -eq 1 ]; then

    # Start from master
    git checkout master

    # Be sure your working branch is clean
    if [ "$(git status --porcelain)" ]; then 
      echo "Please, clean your working directory first."
      exit 1
    else 
      echo "Generating documentation, updating website"; 
    fi

# Check version of the code seen by pdoc
python <<END
import precession
print "Python module precession, version", precession.__version__
END

    # Generate documentation using pdc
    pdoc --html --overwrite precession
    # Get rid of precompiled files
    rm precession/*pyc precession/*/*pyc

    # Generate readme (markdown)
    echo "Generating readme"
python <<END
import precession
docs=precession.__doc__                 # Get code docstrings
title="precession\n"+\
      "==========\n\n"+\
      docs                              # Prepend title
splits=title.split('###')               # Separate parts
removed = splits[:2] + splits[3 :]      # Get rid of some details
joined= "###".join(removed)             # Put parts back together
outfilesave = open("README.md","w",0)   # Write to file
outfilesave.write(joined)
outfilesave.close()
END

    # Convert readme to rst (not committed)
    pandoc README.md --from markdown --to rst -s -o README.rst

    # Commit new html to master branch
    git add precession/index.html precession/test/index.html README.md
    git commit -m "Automatic commit from generate_documentation.sh"
    git push

    # Move html files somewhere else
    temp1=`mktemp`
    cp precession/index.html $temp1
    temp2=`mktemp`
    cp precession/test/index.html $temp2

    # Move html files to gh-pages branch (directories there should exist)
    git checkout gh-pages
    mv $temp1 index.html
    mv $temp2 test/index.html

    # Commit new html to gh-pages branch
    git add index.html test/index.html
    git commit -m "Automatic commit from generate_documentation.sh"
    git push

    # Get rid of temp files
    rm -f $temp1 $temp2

    # Back to master
    git checkout master

else
    echo "Generating documentation, local only"; 
    
# Check version of the code seen by pdoc
python <<END
import precession
print "Python module precession, version", precession.__version__
END

    # Generate documentation using pdc
    pdoc --html --overwrite precession
    # Get rid of precompiled files
    rm precession/*pyc precession/*/*pyc

    # Generate readme (markdown)
    echo "Generating readme"
python <<END
import precession
docs=precession.__doc__                 # Get code docstrings
title="precession\n"+\
      "==========\n\n"+\
      docs                              # Prepend title
splits=title.split('###')               # Separate parts
removed = splits[:2] + splits[3 :]      # Get rid of some details
joined= "###".join(removed)             # Put parts back together
outfilesave = open("README.md","w",0)   # Write to file
outfilesave.write(joined)
outfilesave.close()
END

    # Convert readme to rst (not committed)
    pandoc README.md --from markdown --to rst -s -o README.rst


fi

