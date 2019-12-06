#!/usr/bin/env bash
# e.g.
#   bash docs/get_sysinfo.sh

OUTFILE=sysinfo.txt
[ -n "$1" ] && OUTFILE=$1

# https://github.com/dylanaraps/neofetch
neofetch --stdout --color_blocks off > $OUTFILE
