#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching/detail/matchers.hpp>

#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d/nonfree.hpp>
#endif

#include "common/logger.h"

using namespace std;
using namespace cv;

void findFeatures(const Ptr<Feature2D> &finder,
    const vector<Mat> &images,
    vector<detail::ImageFeatures> *features,
    const string &features_desc,
    TimingLogger *logger = nullptr) {
  int images_n = images.size();

  if (logger) logger->Reset(features_desc);
  LOG(INFO) << features_desc;
  for (int i = 0; i < images_n; i++) {
    computeImageFeatures(finder, images[i], (*features)[i]);
    (*features)[i].img_idx = i;
    auto image_name = (stringstream() << "image #" << (i+1) << ", "
        << images[i].cols << "x" << images[i].rows).str();
    LOG(INFO) << "  " << image_name << ": " << (*features)[i].keypoints.size();
    if (logger) logger->AddSplit(image_name);
  }
  // computeImageFeatures(finder, images, *features);
  // logger->AddSplit("find");
  if (logger) logger->DumpToLog();
}

int main(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;
  auto logger = TimingLogger::Create("Images");
  vector<Mat> images{
    imread(samples::findFile(MY_DATA "/stitching/boat1.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/newspaper1.jpg")),
  };
  logger->AddSplit("read");
  logger->DumpToLog();

  int images_n = images.size();
  vector<detail::ImageFeatures> features(images_n);

  vector<pair<Ptr<Feature2D>, string>> finders{
    {ORB::create(), "ORB Features"},
    {AKAZE::create(), "AKAZE Features"},
#ifdef HAVE_OPENCV_XFEATURES2D
    {xfeatures2d::SURF::create(), "SURF Features"},
    {xfeatures2d::SIFT::create(), "SIFT Features"},
#endif
  };

  for (auto finder : finders) {
    cout << endl;
    findFeatures(finder.first, images, &features, finder.second, logger.get());

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
  }

  return 0;
}
