#!/bin/bash

p=$(dirname "$(realpath "$0")")
installation_paths=(
  "$p/tritonserver"
)

installation_files=(
  "$p/tritonserver/triton.sh"
)   

if [ "$1" == "path" ]; then
  for dir in "${installation_paths[@]}"; do
    export PATH=$PATH:$dir
  done 
  echo "Temporary installation: Paths added to PATH environment variable."
elif [ "$1" == "link" ]; then
  for file in $installation_files; do
    if [ -f "$file" ]; then
      if sudo ln -s "$file" /usr/bin/$(basename "$file")
      then
        echo "Symlink created for $file."
      fi
    else
      echo "File $file not found."
    fi
  done
else
  echo "Usage: $0 {path|link}"
  echo "Please specify whether you want to install the scripts with the PATH or symlinks."
  echo ""
  echo "  path: (temporary) Add paths to PATH environment variable, only exists for the lifetime of this shell."
  echo "  link: (permanent) Add symlinks to /usr/bin. Requires sudo."
  exit 1
fi