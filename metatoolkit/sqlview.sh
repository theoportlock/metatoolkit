db=$1
out=$2
inname=$(readlink -f $db)
outname=$(pwd)/$out

schemacrawler \
	--server=sqlite \
	--database=$inname \
	--command=schema \
	--output-format=pdf \
	--info-level=standard \
	--output-file=$outname
