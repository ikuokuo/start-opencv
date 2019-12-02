#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching/detail/matchers.hpp>

#include "common/logger.h"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;
  auto logger = TimingLogger::Create("Features finding");

  vector<Mat> images{
    imread(samples::findFile(MY_DATA "/stitching/boat1.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/newspaper1.jpg")),
  };
  logger->AddSplit("read");

  int images_n = images.size();
  vector<detail::ImageFeatures> features(images_n);

  auto finder = ORB::create();

  for (int i = 0; i < images_n; i++) {
    computeImageFeatures(finder, images[i], features[i]);
    features[i].img_idx = i;
    auto image_name = (stringstream() << "image #" << (i+1) << ", "
        << images[i].cols << "x" << images[i].rows).str();
    LOG(INFO) << "Features in " << image_name << ": " << features[i].keypoints.size();
    logger->AddSplit(image_name);
  }
  // computeImageFeatures(finder, images, features);
  // logger->AddSplit("find");

  logger->DumpToLog();

  // show

  for (int i = 0; i < images_n; i++) {
    auto img = images[i];
    drawKeypoints(img, features[i].keypoints, img, Scalar(0, 255, 0),
        DrawMatchesFlags::DRAW_OVER_OUTIMG);
    imshow((stringstream() << "image #" << (i+1) << ", "
        << img.cols << "x" << img.rows).str(), img);
  }
  waitKey(0);
  destroyAllWindows();

  return 0;
}
