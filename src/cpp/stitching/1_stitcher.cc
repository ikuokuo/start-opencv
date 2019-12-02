#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/stitching.hpp>

#include "common/logger.h"

using namespace std;
using namespace cv;

int stitch(Stitcher::Mode mode, vector<Mat> images, Mat *pano) {
  auto stitcher = Stitcher::create(mode);
  auto status = stitcher->stitch(images, *pano);
  if (status != Stitcher::OK) {
    LOG(ERROR) << "Can't stitch images, error code = " << int(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int main(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;

  // panorama
  auto logger = TimingLogger::Create("Stitching, boat6");

  Mat pano1;
  vector<Mat> images1{
    imread(samples::findFile(MY_DATA "/stitching/boat1.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/boat2.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/boat3.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/boat4.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/boat5.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/boat6.jpg")),
  };
  logger->AddSplit("read");
  auto ret = stitch(Stitcher::PANORAMA, images1, &pano1);
  logger->AddSplit("stitch");
  if (ret) return EXIT_FAILURE;
  logger->DumpToLog();

  // scans
  logger->Reset("Stitching, newspaper4");

  Mat pano2;
  vector<Mat> images2{
    imread(samples::findFile(MY_DATA "/stitching/newspaper1.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/newspaper2.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/newspaper3.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/newspaper4.jpg")),
  };
  logger->AddSplit("read");
  ret = stitch(Stitcher::SCANS, images2, &pano2);
  logger->AddSplit("stitch");
  if (ret) return EXIT_FAILURE;
  logger->DumpToLog();

  // show

  imshow("pano1", pano1);
  imshow("pano2", pano2);
  waitKey(0);
  destroyAllWindows();

  return EXIT_SUCCESS;
}
