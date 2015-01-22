#!/bin/bash

git status -s | grep -E "\.cpp$|\.h$" | cut -c4- | xargs astyle --indent=spaces=4 --convert-tabs --pad-oper --suffix=none
