#!/usr/bin/env bash
# e.g.
#   bash docs/get_output.sh -c cpp_cmds_stitching.sh -o output/cpp/stitching
#   bash docs/get_output.sh -c cpp_cmds_stitching_try_cuda.sh -o output/cpp/stitching_try_cuda

BASE_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(realpath "$BASE_DIR/..")

ECHO="echo -e"
DATE="date"
# brew install coreutils
[[ "$OSTYPE" == "darwin"* ]] && DATE="gdate"

# options

USAGE="Usage: bash get_output.sh -c <cmds_path> -o <out_dir>\n
  e.g. bash get_output.sh -c cpp_cmds_stitching.sh -o output/cpp/stitching"

while getopts ":c:o:h" opt; do
  case $opt in
    c) cmds_path=$OPTARG;;
    o) out_dir=$OPTARG;;
    h) $ECHO $USAGE; exit;;
    :) $ECHO "Error: option '-$OPTARG' requires an argument" >&2
       $ECHO $USAGE; exit 2;;
    ?) $ECHO "Error: option '-$OPTARG' is illegal" >&2
       $ECHO $USAGE; exit 2;;
  esac
done
[ $# -le 0 ] && $ECHO "Error: options are required :(" >&2 && $ECHO $USAGE && exit 2
[ -z $cmds_path ] && $ECHO "Error: option '-c' is required" >&2 && exit 2
[ -z $out_dir ] && $ECHO "Error: option '-o' is required" >&2 && exit 2

# init

cmds_path_real=$BASE_DIR/$cmds_path
out_dir_real=$BASE_DIR/$out_dir

if [ ! -e $cmds_path_real ]; then
  $ECHO "Error: cmds_path not exists, $cmds_path_real" >&2
  exit 2
fi
if [ ! -d $out_dir_real ]; then
  mkdir -p $out_dir_real
fi

source $cmds_path_real

# run

cd $ROOT_DIR

for cmd in "${CMDS[@]}"; do
  t_start=$($DATE +%s%3N)
  cmd_name=${cmd/ --*/}
  cmd_name=${cmd_name/*\//}
  out_path=$out_dir/$cmd_name.txt
  out_path_real=$BASE_DIR/$out_path
  $ECHO CMD: $cmd_name
  $ECHO RUN: $cmd
  $ECHO OUT: $out_path
  $ECHO $ $cmd > $out_path_real
  $cmd >> $out_path_real
  $ECHO COST: $(($($DATE +%s%3N) - ${t_start})) ms
  $ECHO
done
