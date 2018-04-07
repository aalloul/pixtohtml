#!/usr/bin/env bash
echo "clone"
git clone https://adampackets:e2XKKZho3bc8@github.com/adampackets/pixtohtml.git
cd pixtohtml
mkdir data
echo "cd data"
cd data
echo "download... "
wget  --output-document=data_set.tgz $1
echo "untar..."
tar xf data_set.tgz
echo "Done"
cd ..