#include <algorithm>
#include <fstream>
#include <regex>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>

#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d/nonfree.hpp>
#endif

#define LOG_V 1
#include "common/logger.h"
#include "common/optparse.h"

using namespace std;
using namespace cv;

void checkValue(const string &value, const string &pattern, char delim = '|');
void checkMatch(const string &value, const string &pattern);

int main(int argc, char const *argv[]) {
  // options
  auto parser = optparse::OptionParser()
      .usage("usage: %prog [options]\n"
             "       %prog --work_megapix 0.6 --features orb --matcher homography --match_conf 0.3 --range_width -1 --conf_thresh 1 --save_graph graph.txt --estimator homography --ba ray --ba_refine_mask xxxxx --wave_correct horiz")
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
  group_m.add_option("--conf_thresh").dest("conf_thresh")
      .type("float").set_default(1.0)
      .help("Threshold for two images are from the same panorama confidence. The default is %default.");
  group_m.add_option("--save_graph").dest("save_graph")
      .type("string").metavar("FILENAME")
      .help("Save matches graph represented in DOT language to a file."
            "Labels description: Nm is number of matches, Ni is number of inliers, C is confidence.");
  group_m.add_option("--estimator").dest("estimator_type")
      .type("string").set_default("homography")
      .metavar("homography|affine")
      .help("Type of estimator used for transformation estimation. The default is '%default'.");
  group_m.add_option("--ba").dest("ba_cost_func")
      .type("string").set_default("ray")
      .metavar("no|reproj|ray|affine")
      .help("Bundle adjustment cost function. The default is '%default'.");
  group_m.add_option("--ba_refine_mask").dest("ba_refine_mask")
      .type("string").set_default("xxxxx")
      .metavar("x_xxx")
      .help("Set refinement mask for bundle adjustment. It looks like 'x_xxx',\n"
            "where 'x' means refine respective parameter and '_' means don't\n"
            "refine one, and has the following format:\n"
            "<fx><skew><ppx><aspect><ppy>. The default mask is '%default'. If bundle\n"
            "adjustment doesn't support estimation of selected parameter then\n"
            "the respective flag is ignored.");
  group_m.add_option("--wave_correct").dest("wave_correct")
      .type("string").set_default("horiz")
      .metavar("no|horiz|vert")
      .help("Perform wave effect correction. The default is '%default'.");
  parser.add_option_group(group_m);

  auto options = parser.parse_args(argc, argv);
  bool preview = options.get("preview");
  bool try_cuda = options.get("try_cuda");
  float work_megapix = options.get("work_megapix");
  string features_type = options["features_type"];
  string matcher_type = options["matcher_type"];
  float match_conf = options.get("match_conf");
  int range_width = options.get("range_width");
  float conf_thresh = options.get("conf_thresh");
  string save_graph = options["save_graph"];
  string estimator_type = options["estimator_type"];
  string ba_cost_func = options["ba_cost_func"];
  string ba_refine_mask = options["ba_refine_mask"];
  string wave_correct = options["wave_correct"];
  {
    checkValue(features_type, "surf|orb|sift|akaze");
    checkValue(matcher_type, "homography|affine");
    if (!options.is_set("match_conf")) {
      if (features_type == "orb") match_conf = 0.3;
      else if (features_type == "surf") match_conf = 0.65;
    }
    checkValue(estimator_type, "homography|affine");
    checkValue(ba_cost_func, "no|reproj|ray|affine");
    checkMatch(ba_refine_mask, "[_x]{5}");
    checkValue(wave_correct, "no|horiz|vert");

    LOG(INFO) << "Options:" << endl
        << "  preview: " << (preview ? "true" : "false") << endl
        << "  try_cuda: " << (try_cuda ? "true" : "false") << endl
        << endl
        << "  work_megapix: " << work_megapix << endl
        << "  features: " << features_type << endl
        << "  matcher: " << matcher_type << endl
        << "  match_conf: " << match_conf << endl
        << "  range_width: " << range_width << endl
        << "  conf_thresh: " << conf_thresh << endl
        << "  save_graph: " << save_graph << endl
        << "  estimator: " << estimator_type << endl
        << "  ba: " << ba_cost_func << endl
        << "  ba_refine_mask: " << ba_refine_mask << endl
        << "  wave_correct: " << wave_correct << endl;
  }

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
    if (VLOG_IS_ON(2)) {
      for (auto info : pairwise_matches) {
        VLOG(2) << "  " << info.src_img_idx << " > " << info.dst_img_idx
            << ", conf: " << info.confidence;
      }
    }
  }
  logger->DumpToLog();
  LOG(INFO);

  // rotation estimation
  logger->Reset("Rotation Estimation");
  {
    // check if we should save matches graph
    if (!save_graph.empty()) {
      LOG(INFO) << "Saving matches graph: " << save_graph;
      ofstream f(save_graph);
      f << detail::matchesGraphAsString(images_name, pairwise_matches, conf_thresh);
      logger->AddSplit("save graph");
    }

    // leave only images we are sure are from the same panorama
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    {
      vector<Mat> img_subset;
      vector<String> img_name_subset;
      vector<Size> img_full_size_subset;
      for (size_t i = 0; i < indices.size(); ++i) {
        auto indice = indices[i];
        img_name_subset.push_back(images_name[indice]);
        img_subset.push_back(images[indice]);
        img_full_size_subset.push_back(images_full_size[indice]);
      }

      images = img_subset;
      images_name = img_name_subset;
      images_full_size = img_full_size_subset;

      // check if we still have enough images
      images_n = static_cast<int>(images_name.size());
      if (images_n < 2) {
        LOG(FATAL) << "Need more images";
        return 1;
      }
      LOG(INFO) << "Leave images: " << images_n;
      logger->AddSplit("leave images");
    }

    // estimate camera parameters
    vector<detail::CameraParams> cameras;
    {
      Ptr<detail::Estimator> estimator;
      if (estimator_type == "affine") {
        estimator = makePtr<detail::AffineBasedEstimator>();
      } else {
        estimator = makePtr<detail::HomographyBasedEstimator>();
      }

      if (!(*estimator)(features, pairwise_matches, cameras)) {
        LOG(FATAL) << "Estimate camera parameters failed";
        return 1;
      }
      logger->AddSplit("estimate params");

      for (size_t i = 0; i < cameras.size(); ++i) {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        VLOG(1) << "Initial camera intrinsics #" << indices[i]+1
            << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R;
      }
      logger->AddSplit("convert params");
    }

    // refine camera parameters
    {
      Ptr<detail::BundleAdjusterBase> adjuster;
      if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
      else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
      else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
      else if (ba_cost_func == "no") adjuster = makePtr<detail::NoBundleAdjuster>();
      else {  // NOLINT
        LOG(FATAL) << "Unknown bundle adjustment cost function: " << ba_cost_func;
        return 2;
      }
      adjuster->setConfThresh(conf_thresh);
      Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
      if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
      if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
      if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
      if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
      if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
      adjuster->setRefinementMask(refine_mask);
      if (!(*adjuster)(features, pairwise_matches, cameras)) {
          LOG(FATAL) << "Refine camera parameters failed";
          return 1;
      }
      logger->AddSplit("refine params");
    }

    // find median focal length
    float warped_image_scale;
    {
      vector<double> focals;
      for (size_t i = 0; i < cameras.size(); ++i) {
        VLOG(1) << "Camera #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R;
        focals.push_back(cameras[i].focal);
      }

      sort(focals.begin(), focals.end());
      if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
      else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

      logger->AddSplit("find focals");
    }

    if (wave_correct != "no") {
      auto kind = detail::WAVE_CORRECT_HORIZ;
      if (wave_correct == "vert")
        kind = detail::WAVE_CORRECT_VERT;

      vector<Mat> rmats;
      for (size_t i = 0; i < cameras.size(); ++i)
        rmats.push_back(cameras[i].R.clone());
      detail::waveCorrect(rmats, kind);
      for (size_t i = 0; i < cameras.size(); ++i)
        cameras[i].R = rmats[i];

      logger->AddSplit("wave correct");
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

void checkMatch(const string &value, const string &pattern) {
  regex re(pattern);
  if (!regex_match(value, re)) {
    LOG(FATAL) << "Bad option value: " << value << " (" << pattern << ")";
  }
}
