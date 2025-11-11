#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    curl -fsSL https://apt.llvm.org/llvm.sh | sudo bash -s -- 21
    sudo apt update
    sudo apt install llvm-21
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install llvm@21
else
    exit 1
fi