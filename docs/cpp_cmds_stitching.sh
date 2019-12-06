CMDS_PREFIX=./_output/bin/stitching
CMDS_OUTDIR=docs/output/cpp/stitching
CMDS=(
$CMDS_PREFIX/2_features_finding
$CMDS_PREFIX/2_features_finding2
$CMDS_PREFIX/3_images_matching
$CMDS_PREFIX/4_rotation_estimation
$CMDS_PREFIX/5_images_warping
$CMDS_PREFIX/6_exposure_compensation
$CMDS_PREFIX/7_seam_estimation
"$CMDS_PREFIX/8_image_blenders --output $CMDS_OUTDIR/8_image_blenders.jpg"
"$CMDS_PREFIX/8_image_blenders2 --work_megapix 1 --features orb --matcher affine --match_conf 0.3 --conf_thresh 0.3 --estimator affine --ba affine --ba_refine_mask xxxxx --wave_correct no --warp affine --output $CMDS_OUTDIR/8_image_blenders2.jpg"
)
