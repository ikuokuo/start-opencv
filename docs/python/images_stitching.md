# Images stitching

## Offical Tutorials

### stitching

Boat,

```bash
python src/python/stitching/tutorial/stitching.py --mode 0 data/stitching/boat*
```

Newspaper,

```bash
python src/python/stitching/tutorial/stitching.py --mode 1 data/stitching/newspaper*
```

### stitching_detailed

Boat,

```bash
export IMGS="data/stitching/boat5.jpg \
data/stitching/boat2.jpg \
data/stitching/boat3.jpg \
data/stitching/boat4.jpg \
data/stitching/boat1.jpg \
data/stitching/boat6.jpg"

python src/python/stitching/tutorial/stitching_detailed.py $IMGS --work_megapix 0.6 --features orb --matcher homography --estimator homography --match_conf 0.3 --conf_thresh 0.3 --ba ray --ba_refine_mask xxxxx --save_graph test.txt --wave_correct no --warp fisheye --blend multiband --expos_comp no --seam gc_colorgrad

python src/python/stitching/tutorial/stitching_detailed.py $IMGS --work_megapix 0.6 --features orb --matcher homography --estimator homography --match_conf 0.3 --conf_thresh 0.3 --ba ray --ba_refine_mask xxxxx --wave_correct horiz --warp compressedPlaneA2B1 --blend multiband --expos_comp channel_blocks --seam gc_colorgrad
```

Newspaper,

```bash
export IMGS="data/stitching/newspaper1.jpg \
data/stitching/newspaper2.jpg"

python src/python/stitching/tutorial/stitching_detailed.py $IMGS --work_megapix 0.6 --features surf --matcher affine --estimator affine --match_conf 0.3 --conf_thresh 0.3 --ba affine --ba_refine_mask xxxxx --wave_correct no --warp affine
```

### References

* https://docs.opencv.org/master/d1/d46/group__stitching.html
* https://docs.opencv.org/master/d0/d33/tutorial_table_of_content_stitching.html
