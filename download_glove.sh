curdir=`pwd`
cd ..
mkdir glove
cd glove
# need to download glove from http://nlp.stanford.edu/data/glove.6B.zip
get http://nlp.stanford.edu/data/glove.6B.zip
unzip http://nlp.stanford.edu/data/glove.6B.zip
cd $curdir