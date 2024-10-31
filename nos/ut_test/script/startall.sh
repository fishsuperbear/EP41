#!/bin/bash

num=$1
ut_list=$(ls ut)

export LD_LIBRARY_PATH=/app/lib:$LD_LIBRARY_PATH

for i in $ut_list
do
        cd ut/"$i" || exit
        binary=$(ls bin)
        bash start.sh "$binary" "$num"
        cd - || exit
done