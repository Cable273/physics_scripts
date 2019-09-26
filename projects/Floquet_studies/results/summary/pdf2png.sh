#!/usr/bin/env sh

files=$(ls | grep "pxp")
for file in $files ; do
    filename=$(echo $file | awk -F '.pdf' '{print $1}')
    convert -density 150 -quality 100 $file $file.png
done
