#include <algorithm>
#include <fstream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching/detail/matchers.hpp>

#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d/nonfree.hpp>
#endif

#define LOG_V 1
#include "common/logger.h"
#include "common/optparse.h"

using namespace std;
using namespace cv;

void checkValue(const string &value, const string &pattern, char delim = '|');

int main(int argc, char const *argv[]) {
  // options
  auto parser = optparse::OptionParser()
      .usage("usage: %prog [options]\n"
             "       %prog --work_megapix 0.6 --features orb --matcher homography --match_conf 0.3 --range_width -1")
      .description("Pairwise matching");
  parser.add_option("-p", "--preview").dest("preview")
      .action("store_true").help("Preview features in input images");
  parser.add_option("--try_cuda").dest("try_cuda")
      .action("store_true").help("Try to use CUDA. All default values are for CPU mode.");

  auto group_m = optparse::OptionGroup("Motion Estimation");
  group_m.add_option("--work_megapix").dest("work_megapix")
      .type("float").set_default(0.6)
      .help("Resolution for image registration step. The default is %default Mpx.");
  group_m.add_option("--features").dest("features_type")
      .type("string").set_default("orb")
      .metavar("surf|orb|sift|akaze")
      .help("Type of features used for images matching. The default is '%default'.");
  group_m.add_option("--matcher").dest("matcher_type")
      .type("string").set_default("homography")
      .metavar("homography|affine")
      .help("Matcher used for pairwise image matching. The default is '%default'.");
  group_m.add_option("--match_conf").dest("match_conf")
      .type("float")
      .help("Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.");
  group_m.add_option("--range_width").dest("range_width")
      .type("int").set_default(-1)
      .help("Limit number of images to match with.");
  parser.add_option_group(group_m);

  auto options = parser.parse_args(argc, argv);
  bool preview = options.get("preview");
  bool try_cuda = options.get("try_cuda");
  float work_megapix = options.get("work_megapix");
  string features_type = options["features_type"];
  string matcher_type = options["matcher_type"];
  float match_conf = options.get("match_conf");
  int range_width = options.get("range_width");

  checkValue(features_type, "surf|orb|sift|akaze");
  checkValue(matcher_type, "homography|affine");
  if (!options.is_set("match_conf")) {
    if (features_type == "orb") match_conf = 0.3;
    else if (features_type == "surf") match_conf = 0.65;
  }

  LOG(INFO) << "Options:" << endl
      << "  preview: " << (preview ? "true" : "false") << endl
      << "  try_cuda: " << (try_cuda ? "true" : "false") << endl
      << endl
      << "  work_megapix: " << work_megapix << endl
      << "  features: " << features_type << endl
      << "  matcher: " << matcher_type << endl
      << "  match_conf: " << match_conf << endl
      << "  range_width: " << range_width << endl;

  // images (same size)
  auto logger = TimingLogger::Create("Images");
  vector<Mat> images;
  vector<string> images_name{
    MY_DATA "/stitching/boat5.jpg",
    MY_DATA "/stitching/boat2.jpg",
    MY_DATA "/stitching/boat3.jpg",
    MY_DATA "/stitching/boat4.jpg",
    MY_DATA "/stitching/boat1.jpg",
    MY_DATA "/stitching/boat6.jpg",
  };
  vector<Size> images_full_size;
  {
    // images read
    vector<Mat> images_full;
    for (auto name : images_name) {
      auto img_full = imread(samples::findFile(name));
      images_full.push_back(img_full);
      images_full_size.push_back(img_full.size());
    }
    logger->AddSplit("read");

    double work_scale = 1;
    bool is_work_scale_set = false;

    // images scale
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
        images.push_back(img.clone());
      }
    }
    logger->AddSplit("scale");
    images_full.clear();
  }
  logger->DumpToLog();
  LOG(INFO);

  // features finding
  int images_n = images.size();
  vector<detail::ImageFeatures> features(images_n);
  {
    Ptr<Feature2D> finder;
    string features_desc;
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

    // features compute
    logger->Reset(features_desc);
    {
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
    }
    logger->DumpToLog();

    // features preview
    if (preview) {
      Mat img;
      for (int i = 0; i < images_n; i++) {
        drawKeypoints(images[i], features[i].keypoints, img, Scalar(0, 255, 0),
            DrawMatchesFlags::DEFAULT);
        auto ss = stringstream();
        ss << "image #" << (i+1) << ", " << images[i].cols << "x" << images[i].rows;
        imshow(ss.str(), img);
        waitKey(0);
      }
      destroyAllWindows();
    }
  }
  LOG(INFO);

  // pairwise matching
  logger->Reset("Pairwise Matching");
  vector<detail::MatchesInfo> pairwise_matches;
  {
    Ptr<detail::FeaturesMatcher> matcher;
    if (matcher_type == "affine") {
      matcher = makePtr<detail::AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
    } else if (range_width == -1) {
      matcher = makePtr<detail::BestOf2NearestMatcher>(try_cuda, match_conf);
    } else {
      matcher = makePtr<detail::BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);
    }

    (*matcher)(features, pairwise_matches);
    matcher->collectGarbage();
    logger->AddSplit("match");

    LOG(INFO) << "Pairwise matches: " << pairwise_matches.size();
    if (VLOG_IS_ON(1)) {
      for (auto info : pairwise_matches) {
        VLOG(1) << "  " << info.src_img_idx << " > " << info.dst_img_idx
            << ", conf: " << info.confidence;
      }
    }
  }
  logger->DumpToLog();

  return 0;
}

void checkValue(const string &value, const string &pattern, char delim) {
  vector<string> tokens;
  stringstream ss(pattern);
  string token;
  while (getline(ss, token, delim)) {
    tokens.push_back(token);
  }
  if (find(begin(tokens), end(tokens), value) == end(tokens)) {
    LOG(FATAL) << "Bad option value: " << value << " (" << pattern << ")";
  }
}
