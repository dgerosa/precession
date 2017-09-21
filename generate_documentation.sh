# Usage: bash generate_documentation.sh [-h||--help] [-doc] [-web] [-readme] [-all]
# Generate API documentation from he python's docstrings present in the code using pdoc. Generate readme for github and pypi.

web=0
doc=0
readme=0
while test $# -gt 0; do
        case "$1" in
                -h|--help)
                        echo "Usage: bash generate_documentation.sh -doc -web -readme"
                        echo " "
                        echo "Generate automatic html documentation using pdoc"
                        echo "pdoc is available here https://github.com/BurntSushi/pdoc"
                        echo "and can be installed from Pypi: #pip install pdoc."
                        echo "If -doc, documentation is only produced but not published online"
                        echo "If -web, documentation is pushed to a dedicated git branch (gh-pages)"
                        echo "and published at http://dgerosa.github.io/precession/"
                        echo "If -readme, create a md readme from docstrings and convert it to rst"
                        echo " "
                        echo "options:"
                        echo "   -h, --help       show brief help"
                        echo "   -doc             produce documentation"
                        echo "   -web             produce and publish  documentation"
                        echo "   -readme          produce readme"
                        echo "   -all             do all the previous"

                        exit 0
                        ;;
                -doc)
                        shift
                        doc=1
                        ;;
                -web)
                        shift
                        web=1
                        ;;
                -readme)
                        shift
                        readme=1
                        ;;
                -all)
                        shift
                        doc=1
                        web=1
                        readme=1
                        ;;
                *)
                shift
        esac
done


###################################

if [ $web -eq 0 ] && [ $doc -eq 0 ] && [ $readme -eq 0 ]; then
    echo "Usage: bash generate_documentation.sh [-h||--help] [-doc] [-web] [-readme] [-all]"
    exit 0
fi



# Be sure your working branch is clean
if [ "$(git status --porcelain)" ]; then
    echo "Please, clean your working directory first."
    #exit 1
fi

#pip uninstall -y precession

###################################

if [ $web -eq 1 ]; then

    echo " "
    echo "Generating documentation, updating website"

     # Where you start from
    start=$(pwd)

    # Start from master
    git checkout master

    # Build temporary directory
    mkdir ${HOME}/temp_precession
    mkdir ${HOME}/temp_precession/precession
    mkdir ${HOME}/temp_precession/precession/test
    # Copy code in temp directory
    cp precession/precession.py ${HOME}/temp_precession/precession/__init__.py
    cp precession/test/test.py ${HOME}/temp_precession/precession/test/__init__.py
    cp setup.py ${HOME}/temp_precession/setup.py
    cp README.rst ${HOME}/temp_precession/README.rst

    # Go there
    cd ${HOME}/temp_precession
    #python setup.py install

    # Check version of the code seen by pdoc
python <<END
import precession
print "Python module precession, version", precession.__version__
END

    # Generate documentation using pdc
    pdoc --html --overwrite precession
    #pip uninstall -y precession

    # Go back
    cd ${start}

    # Move html files to gh-pages branch (directories there should exist)
    git checkout gh-pages
    mv ${HOME}/temp_precession/precession/index.html index.html
    mv ${HOME}/temp_precession/precession/test/index.html test/index.html

    # Commit new html to gh-pages branch
    git add index.html test/index.html
    git commit -m "Automatic commit from generate_documentation.sh"
    git push

    # Back to master
    git checkout master

    # Get rid of temp files
    rm -rf ${HOME}/temp_precession

fi

###################################

if [ $doc -eq 1 ]; then

    echo " "
    echo "Generating documentation, local version"

    # Where you start from
    start=$(pwd)

    # Start from master
    #git checkout master

    # Build temporary directory
    mkdir ${HOME}/temp_precession
    mkdir ${HOME}/temp_precession/precession
    mkdir ${HOME}/temp_precession/precession/test
    # Copy code in temp directory
    cp precession/precession.py ${HOME}/temp_precession/precession/__init__.py
    cp precession/test/test.py ${HOME}/temp_precession/precession/test/__init__.py
    cp setup.py ${HOME}/temp_precession/setup.py
    cp README.rst ${HOME}/temp_precession/README.rst

    # Go there
    cd ${HOME}/temp_precession
    #python setup.py install

    # Check version of the code seen by pdoc
    python <<END
import precession
print "Python module precession, version", precession.__version__
END

    # Generate documentation using pdc
    pdoc --html --overwrite precession
    #pip uninstall -y precession

    # Go back
    cd ${start}

    mv ${HOME}/temp_precession/precession/index.html precession/index.html
    mv ${HOME}/temp_precession/precession/test/index.html precession/test/index.html


    # Commit new html to master branch
    git add precession/index.html precession/test/index.html
    git commit -m "Automatic commit from generate_documentation.sh"
    git push

    # rm pyc files
    rm -rf precession/__init__.pyc precession/test/__init__.pyc

    # Get rid of temp files
    rm -rf ${HOME}/temp_precession


fi

###################################


if [ $readme -eq 1 ]; then

    echo " "
    echo "Generating readme"

    # Where you start from
    start=$(pwd)

    # Build temporary directory
    mkdir ${HOME}/temp_precession
    mkdir ${HOME}/temp_precession/precession
    mkdir ${HOME}/temp_precession/precession/test
    # Copy code in temp directory
    cp precession/precession.py ${HOME}/temp_precession/precession/__init__.py
    cp precession/test/test.py ${HOME}/temp_precession/precession/test/__init__.py
    cp setup.py ${HOME}/temp_precession/setup.py
    cp README.rst ${HOME}/temp_precession/README.rst

    # Go there
    cd ${HOME}/temp_precession
    #python setup.py install

    # Generate readme in markdown using python's docstrings
python <<END
import precession
docs=precession.__doc__                 # Get code docstrings
title="precession\n"+\
      "==========\n\n"+\
      docs                              # Prepend title
splits=title.split('###')               # Separate parts
removed = splits[:3] + splits[4 :]      # Get rid of some details
joined= "###".join(removed)             # Put parts back together
outfilesave = open("README.md","w",0)   # Write to file
outfilesave.write(joined)
outfilesave.close()
END
    #pip uninstall -y precession

    # Go back
    cd ${start}

    mv ${HOME}/temp_precession/README.md README.md

    # Convert readme to rst (this is ignored by git, but needed to upload on pypi)
    pandoc README.md --from markdown --to rst -s -o README.rst

    # Commit readme to master branch
    git add README.md README.rst
    git commit -m "Automatic commit from generate_documentation.sh"
    git push

    # rm pyc files
    rm -rf precession/__init__.pyc precession/test/__init__.pyc

    # Get rid of temp files
    rm -rf ${HOME}/temp_precession

fi

#pip install precession
