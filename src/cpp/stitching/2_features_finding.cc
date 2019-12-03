#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching/detail/matchers.hpp>

#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d/nonfree.hpp>
#endif

#include "common/logger.h"
#include "common/optparse.h"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {
  // options
  auto parser = optparse::OptionParser()
      .usage("usage: %prog [options]")
      .description("Features Finding");
  parser.add_option("-p", "--preview").dest("preview")
      .action("store_true").help("Preview features in input images");
  parser.add_option("--work_megapix").dest("work_megapix")
      .type("float").set_default(0.6)
      .help("Resolution for image registration step. The default is %default Mpx.");

  auto options = parser.parse_args(argc, argv);
  bool preview = options.get("preview");
  float work_megapix = options.get("work_megapix");

  LOG(INFO) << "Options:" << endl
      << "  preview: " << (preview ? "true" : "false") << endl
      << "  work_megapix: " << work_megapix << endl;

  // images
  auto logger = TimingLogger::Create("Images");
  vector<Mat> images_full{
    imread(samples::findFile(MY_DATA "/stitching/boat1.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/newspaper1.jpg")),
  };
  logger->AddSplit("read");

  // images scale
  vector<Mat> images;
  if (work_megapix < 0) {
    images = move(images_full);
  } else {
    Mat img;
    for (auto img_full : images_full) {
      double work_scale = min(1.0, sqrt(work_megapix * 1e6 / img_full.size().area()));
      resize(img_full, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
      images.push_back(img);
    }
  }
  logger->AddSplit("scale");
  logger->DumpToLog();
  images_full.clear();

  // features finding

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

  auto findFeatures = [](const Ptr<Feature2D> &finder,
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
      auto ss = stringstream();
      ss << "image #" << (i+1) << ", " << images[i].cols << "x" << images[i].rows;
      auto image_name = ss.str();
      LOG(INFO) << "  " << image_name << ": " << (*features)[i].keypoints.size();
      if (logger) logger->AddSplit(image_name);
    }
    // computeImageFeatures(finder, images, *features);
    // logger->AddSplit("find");
    if (logger) logger->DumpToLog();
  };

  for (auto finder : finders) {
    LOG(INFO);
    findFeatures(finder.first, images, &features, finder.second, logger.get());

    // features preview
    if (preview) {
      Mat img;
      for (int i = 0; i < images_n; i++) {
        drawKeypoints(images[i], features[i].keypoints, img, Scalar(0, 255, 0),
            DrawMatchesFlags::DEFAULT);
        auto ss = stringstream();
        ss << "image #" << (i+1) << ", " << images[i].cols << "x" << images[i].rows;
        imshow(ss.str(), img);
      }
      waitKey(0);
      destroyAllWindows();
    }
  }

  return 0;
}
