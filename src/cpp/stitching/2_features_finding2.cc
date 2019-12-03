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
  parser.add_option("--features").dest("features_type")
      .type("string").set_default("orb")
      .metavar("surf|orb|sift|akaze")
      .help("Type of features used for images matching. The default is %default.");

  auto options = parser.parse_args(argc, argv);
  bool preview = options.get("preview");
  float work_megapix = options.get("work_megapix");
  string features_type = options["features_type"];

  LOG(INFO) << "Options:" << endl
      << "  preview: " << (preview ? "true" : "false") << endl
      << "  work_megapix: " << work_megapix << endl
      << "  features: " << features_type << endl;

  // images (same size)
  auto logger = TimingLogger::Create("Images");
  vector<Mat> images_full{
    imread(samples::findFile(MY_DATA "/stitching/boat1.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/boat2.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/boat3.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/boat4.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/boat5.jpg")),
    imread(samples::findFile(MY_DATA "/stitching/boat6.jpg")),
  };
  logger->AddSplit("read");

  // variables
  double work_scale = 1;
  bool is_work_scale_set = false;

  // images scale
  vector<Mat> images;
  if (work_megapix < 0) {
    images = move(images_full);
    work_scale = 1;
    is_work_scale_set = true;
  } else {
    Mat img;
    for (auto img_full : images_full) {
      if (!is_work_scale_set) {
        work_scale = min(1.0, sqrt(work_megapix * 1e6 / img_full.size().area()));
        is_work_scale_set = true;
      }
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

  Ptr<Feature2D> finder;
  std::string features_desc;
  if (features_type == "orb") {
    finder = ORB::create();
    features_desc = "ORB Features";
  } else if (features_type == "akaze") {
    finder = AKAZE::create();
    features_desc = "AKAZE Features";
  }
#ifdef HAVE_OPENCV_XFEATURES2D
  else if (features_type == "surf") {  // NOLINT
    finder = xfeatures2d::SURF::create();
    features_desc = "SURF Features";
  } else if (features_type == "sift") {
    finder = xfeatures2d::SIFT::create();
    features_desc = "SIFT Features";
  }
#endif
  else {  // NOLINT
    LOG(ERROR) << "Unknown 2D features type: " << features_type;
    return 2;
  }

  LOG(INFO);
  {
    logger->Reset(features_desc);
    LOG(INFO) << features_desc;
    for (int i = 0; i < images_n; i++) {
      computeImageFeatures(finder, images[i], features[i]);
      features[i].img_idx = i;
      auto ss = stringstream();
      ss << "image #" << (i+1) << ", " << images[i].cols << "x" << images[i].rows;
      auto image_name = ss.str();
      LOG(INFO) << "  " << image_name << ": " << features[i].keypoints.size();
      logger->AddSplit(image_name);
    }
    logger->DumpToLog();
  }

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

  return 0;
}
