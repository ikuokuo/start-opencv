CMDS_PREFIX=./_output/bin/stitching
CMDS_OUTDIR=docs/output/cpp/stitching_try_cuda
CMDS=(
"$CMDS_PREFIX/8_image_blenders --try_cuda --output $CMDS_OUTDIR/8_image_blenders.jpg"
"$CMDS_PREFIX/8_image_blenders2 --try_cuda --work_megapix 1 --features orb --matcher affine --match_conf 0.3 --conf_thresh 0.3 --estimator affine --ba affine --ba_refine_mask xxxxx --wave_correct no --warp affine --output $CMDS_OUTDIR/8_image_blenders2.jpg"
)
