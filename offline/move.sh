#!/bin/bash

BASE_DIR="/data/13_project/sanity_llama_8B"

mkdir -p "$BASE_DIR/maxed"
mkdir -p "$BASE_DIR/summed"

for dir in "$BASE_DIR"/*/; do
    dirname=$(basename "$dir")

    if [[ "$dirname" == "maxed" || "$dirname" == "summed" ]]; then
        continue
    fi

    for file in "$dir"/*.pt; do
        [ -e "$file" ] || continue

        filename=$(basename "$file")
        newname="${dirname}_${filename}"

        if [[ "$filename" == *_maxed.pt ]]; then
            cp "$file" "$BASE_DIR/maxed/$newname"
        elif [[ "$filename" == *_summed.pt ]]; then
            cp "$file" "$BASE_DIR/summed/$newname"
        fi
    done
done