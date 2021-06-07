# Image Processing

## Image Gradients

```bash
python src/python/processing/gradients.py data/processing/sudoku.png
```

## Sobel Derivatives

```bash
python src/python/processing/sobel.py data/processing/sudoku.png
python src/python/processing/scharr.py data/processing/sudoku.png
```

## Laplace Operator

```bash
python src/python/processing/laplace.py data/processing/sudoku.png
```

## Contours

```bash
python src/python/processing/contours.py -i data/processing/dog_catch_ball_thres_ball.png --no-show-boxes
python src/python/processing/contours.py -i data/processing/dog_catch_ball_thres_dog.png --no-show-circles

python src/python/processing/contours.py -i data/processing/dog_catch_ball.jpg -to --thres-hsv -mo --no-show-boxes
```

### References

* [Image Processing](https://docs.opencv.org/master/d7/da8/tutorial_table_of_content_imgproc.html)

* [Image Gradients](https://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html)
* [Sobel Derivatives](https://docs.opencv.org/master/d2/d2c/tutorial_sobel_derivatives.html)
* [Laplace Operator](https://docs.opencv.org/master/d5/db5/tutorial_laplace_operator.html)
