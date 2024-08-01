#!/bin/bash

# Function to rename directories according to their position
rename_directories() {
    local base_path=$1
    local prefix="version_"

    if [ -z "$base_path" ]; then
        echo "Please provide the base path."
        return 1
    fi

    if [ ! -d "$base_path" ]; then
        echo "The provided path is not a directory."
        return 1
    fi

    cd "$base_path" || return 1

    # List directories matching the pattern 'version*', sort them, and rename them
    count=1
    for dir in $(ls -d version* 2>/dev/null | sort); 
    do
        new_name="${prefix}${count}"
        if [ "$dir" != "$new_name" ]; then
            mv "$dir" "$new_name"
        fi
        count=$((count + 1))
    done

    echo "Renaming completed."
}

# Call the function with the base directory as argument
rename_directories "$1"

