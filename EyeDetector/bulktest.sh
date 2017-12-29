#!/bin/bash

for file in img/*
do
    result="./EyeDetector $file"
    printf "%s:%s\n" "$file" " $(eval $result)"
done
