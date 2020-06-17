#include <algorithm>
#include <fstream>
#include <regex>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/timelapsers.hpp>
// #include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/stitching/warpers.hpp>

#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d/nonfree.hpp>
#endif

// #define LOG_V 1
#include "common/logger.h"
#include "common/optparse.h"

using namespace std;
using namespace cv;

void checkValue(const string &value, const string &pattern, char delim = '|');
void checkMatch(const string &value, const string &pattern);

int main(int argc, char const *argv[]) {
  auto applog = TimingLogger::Create("App");

  // options
  auto parser = optparse::OptionParser()
      .usage("usage: %prog [options]\n"
             "       %prog --work_megapix 1 --features orb --matcher affine --match_conf 0.3 --conf_thresh 0.3 --estimator affine --ba affine --ba_refine_mask xxxxx --wave_correct no --warp affine")
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

  auto group_c = optparse::OptionGroup("Compositing");
  group_c.add_option("--warp").dest("warp_type")
      .type("string").set_default("spherical")
      .metavar("affine|plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator")
      .help("Warp surface type. The default is '%default'.");
  group_c.add_option("--seam_megapix").dest("seam_megapix")
      .type("float").set_default(0.1)
      .help("Resolution for seam estimation step. The default is %default Mpx.");
  group_c.add_option("--expos_comp").dest("expos_comp_type")
      .type("string").set_default("gain_blocks")
      .metavar("no|gain|gain_blocks|channels|channels_blocks")
      .help("Exposure compensation method. The default is '%default'.");
  group_c.add_option("--expos_comp_nr_feeds").dest("expos_comp_nr_feeds")
      .type("int").set_default(1)
      .help("Number of exposure compensation feed. The default is %default.");
  group_c.add_option("--expos_comp_nr_filtering").dest("expos_comp_nr_filtering")
      .type("int").set_default(2)
      .help("Number of filtering iterations of the exposure compensation gains.\n"
            "Only used when using a block exposure compensation method.\n"
            "The default is %default.");
  group_c.add_option("--expos_comp_block_size").dest("expos_comp_block_size")
      .type("int").set_default(32)
      .help("BLock size in pixels used by the exposure compensator.\n"
            "Only used when using a block exposure compensation method.\n"
            "The default is %default.");
  group_c.add_option("--seam").dest("seam_find_type")
      .type("string").set_default("gc_color")
      .metavar("no|voronoi|gc_color|gc_colorgrad|dp_color|dp_colorgrad")
      .help("Seam estimation method. The default is '%default'.");
  group_c.add_option("--compose_megapix").dest("compose_megapix")
      .type("float").set_default(-1)
      .help("Resolution for compositing step. Use -1 for original resolution. The default is %default.");
  group_c.add_option("--blend").dest("blend_type")
      .type("string").set_default("multiband")
      .metavar("no|feather|multiband")
      .help("Blending method. The default is '%default'.");
  group_c.add_option("--blend_strength").dest("blend_strength")
      .type("float").set_default(5)
      .help("Blending strength from [0,100] range. The default is %default.");
  group_c.add_option("--timelapse").dest("timelapse_type")
      .type("string").set_default("no")
      .metavar("no|as_is|crop")
      .help("Output warped images separately as frames of a time lapse movie, with 'fixed_' prepended to input file names. The default is '%default'.");
  group_c.add_option("-o", "--output").dest("result_name")
      .type("string").set_default("result.jpg").metavar("result.jpg")
      .help("The default is '%default'.");
  parser.add_option_group(group_c);

  auto options = parser.parse_args(argc, argv);
  bool preview = options.get("preview");
  bool try_cuda = options.get("try_cuda");
  // motion estimation
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
  // compositing
  string warp_type = options["warp_type"];
  float seam_megapix = options.get("seam_megapix");
  string expos_comp_type = options["expos_comp_type"];
  int expos_comp_nr_feeds = options.get("expos_comp_nr_feeds");
  int expos_comp_nr_filtering = options.get("expos_comp_nr_filtering");
  int expos_comp_block_size = options.get("expos_comp_block_size");
  string seam_find_type = options["seam_find_type"];
  float compose_megapix = options.get("compose_megapix");
  string blend_type = options["blend_type"];
  float blend_strength = options.get("blend_strength");
  string timelapse_type = options["timelapse_type"];
  string result_name = options["result_name"];
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

    checkValue(warp_type, "affine|plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator");
    checkValue(expos_comp_type, "no|gain|gain_blocks|channels|channels_blocks");
    checkValue(seam_find_type, "no|voronoi|gc_color|gc_colorgrad|dp_color|dp_colorgrad");
    checkValue(blend_type, "no|feather|multiband");
    checkValue(timelapse_type, "no|as_is|crop");

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
        << "  wave_correct: " << wave_correct << endl
        << endl
        << "  warp: " << warp_type << endl
        << "  seam_megapix: " << seam_megapix << endl
        << "  expos_comp: " << expos_comp_type << endl
        << "  expos_comp_nr_feeds: " << expos_comp_nr_feeds << endl
        << "  expos_comp_nr_filtering: " << expos_comp_nr_filtering << endl
        << "  expos_comp_block_size: " << expos_comp_block_size << endl
        << "  seam_find_type: " << seam_find_type << endl
        << "  compose_megapix: " << compose_megapix << endl
        << "  blend: " << blend_type << endl
        << "  blend_strength: " << blend_strength << endl
        << "  timelapse: " << timelapse_type << endl
        << "  output: " << result_name << endl;
  }
  applog->AddSplit("options parsing");

  // images (same size)
  auto logger = TimingLogger::Create("Images Reading");
  vector<string> images_name{
    MY_DATA "/stitching/newspaper1.jpg",
    MY_DATA "/stitching/newspaper2.jpg",
    MY_DATA "/stitching/newspaper3.jpg",
    MY_DATA "/stitching/newspaper4.jpg",
  };
  vector<Mat> images_full;
  vector<Size> images_full_size;
  {
    // images read
    for (auto name : images_name) {
      auto img_full = imread(samples::findFile(name));
      if (img_full.empty()) {
        LOG(FATAL) << "Can't open image " << name;
        return 1;
      }
      images_full.push_back(img_full);
      images_full_size.push_back(img_full.size());
    }
    logger->AddSplit("read");
  }
  logger->DumpToLog();
  LOG(INFO);
  applog->AddSplit("images reading");

  // features finding
  int images_n = images_name.size();
  vector<Mat> images(images_n);
  vector<detail::ImageFeatures> features(images_n);
  double work_scale = 1, seam_scale = 1;
  double seam_work_aspect = 1;
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
#ifdef HAVE_OPENCV_FEATURES2D_SIFT
    else if (features_type == "sift") {
      finder = SIFT::create();
      features_desc = "SIFT Features";
    }
#endif
#ifdef HAVE_OPENCV_XFEATURES2D
    else if (features_type == "surf") {  // NOLINT
      finder = xfeatures2d::SURF::create();
      features_desc = "SURF Features";
    }
#ifdef HAVE_OPENCV_XFEATURES2D_SIFT
    else if (features_type == "sift") {
      finder = xfeatures2d::SIFT::create();
      features_desc = "SIFT Features";
    }
#endif
#endif
    else {  // NOLINT
      LOG(ERROR) << "Unknown 2D features type: " << features_type;
      return 2;
    }

    bool is_work_scale_set = false, is_seam_scale_set = false;

    // features compute
    logger->Reset(features_desc);
    LOG(INFO) << features_desc;
    Mat img;
    for (int i = 0; i < images_n; i++) {
      auto img_full = images_full[i];

      auto ss = stringstream();
      ss << "image #" << (i+1) << ", " << img_full.cols << "x" << img_full.rows;
      auto img_name = ss.str();

      // work scale
      if (work_megapix < 0) {
        img = img_full;
        work_scale = 1;
        is_work_scale_set = true;
      } else {
        if (!is_work_scale_set) {
          work_scale = min(1.0, sqrt(work_megapix * 1e6 / img_full.size().area()));
          is_work_scale_set = true;
        }
        resize(img_full, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
      }

      if (!is_seam_scale_set) {
        seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / img_full.size().area()));
        seam_work_aspect = seam_scale / work_scale;
        is_seam_scale_set = true;
      }

      // compute
      computeImageFeatures(finder, img, features[i]);
      features[i].img_idx = i;
      LOG(INFO) << "  " << img_name << ": " << features[i].keypoints.size();

      // preview
      if (preview) {
        drawKeypoints(img, features[i].keypoints, img, Scalar(0, 255, 0),
            DrawMatchesFlags::DEFAULT);
        imshow(img_name, img);
        waitKey(0);
      }

      // seam scale
      resize(img_full, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
      images[i] = img.clone();

      logger->AddSplit(img_name);
    }
    img.release();
    logger->DumpToLog();
  }
  LOG(INFO);
  applog->AddSplit("features finding");

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
  applog->AddSplit("pairwise matching");

  // rotation estimation
  logger->Reset("Rotation Estimation");
  vector<int> indices;
  vector<detail::CameraParams> cameras;
  float warped_image_scale;
  {
    // check if we should save matches graph
    if (!save_graph.empty()) {
      LOG(INFO) << "Saving matches graph: " << save_graph;
      ofstream f(save_graph);
      f << detail::matchesGraphAsString(images_name, pairwise_matches, conf_thresh);
      logger->AddSplit("save graph");
    }

    // leave only images we are sure are from the same panorama
    indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
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
  LOG(INFO);
  applog->AddSplit("rotation estimation");

  // images warping
  logger->Reset("Images Warping");
  Ptr<WarperCreator> warper_creator;
  vector<Point> corners(images_n);
  vector<UMat> masks_warped(images_n);
  vector<UMat> images_warped(images_n);
  vector<Size> sizes(images_n);
  vector<UMat> masks(images_n);
  vector<UMat> images_warped_f(images_n);
  {
    // prepare images masks
    for (int i = 0; i < images_n; ++i) {
      masks[i].create(images[i].size(), CV_8U);
      masks[i].setTo(Scalar::all(255));
    }
    logger->AddSplit("prepare masks");

    // warp images and their masks
    if (warp_type == "plane") {
#ifdef HAVE_OPENCV_CUDAWARPING
      if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
        warper_creator = makePtr<PlaneWarperGpu>();
      else
#endif
        warper_creator = makePtr<PlaneWarper>();
    } else if (warp_type == "affine") {
      warper_creator = makePtr<AffineWarper>();
    } else if (warp_type == "cylindrical") {
#ifdef HAVE_OPENCV_CUDAWARPING
      if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
        warper_creator = makePtr<CylindricalWarperGpu>();
      else
#endif
        warper_creator = makePtr<CylindricalWarper>();
    } else if (warp_type == "spherical") {
#ifdef HAVE_OPENCV_CUDAWARPING
      if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
        warper_creator = makePtr<SphericalWarperGpu>();
      else
#endif
        warper_creator = makePtr<SphericalWarper>();
    } else if (warp_type == "fisheye") {
      warper_creator = makePtr<FisheyeWarper>();
    } else if (warp_type == "stereographic") {
      warper_creator = makePtr<StereographicWarper>();
    } else if (warp_type == "compressedPlaneA2B1") {
      warper_creator = makePtr<CompressedRectilinearWarper>(2.0f, 1.0f);
    } else if (warp_type == "compressedPlaneA1.5B1") {
      warper_creator = makePtr<CompressedRectilinearWarper>(1.5f, 1.0f);
    } else if (warp_type == "compressedPlanePortraitA2B1") {
      warper_creator = makePtr<CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
    } else if (warp_type == "compressedPlanePortraitA1.5B1") {
      warper_creator = makePtr<CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
    } else if (warp_type == "paniniA2B1") {
      warper_creator = makePtr<PaniniWarper>(2.0f, 1.0f);
    } else if (warp_type == "paniniA1.5B1") {
      warper_creator = makePtr<PaniniWarper>(1.5f, 1.0f);
    } else if (warp_type == "paniniPortraitA2B1") {
      warper_creator = makePtr<PaniniPortraitWarper>(2.0f, 1.0f);
    } else if (warp_type == "paniniPortraitA1.5B1") {
      warper_creator = makePtr<PaniniPortraitWarper>(1.5f, 1.0f);
    } else if (warp_type == "mercator") {
      warper_creator = makePtr<MercatorWarper>();
    } else if (warp_type == "transverseMercator") {
      warper_creator = makePtr<TransverseMercatorWarper>();
    }
    if (!warper_creator) {
      LOG(FATAL) << "Can't create the following warper '" << warp_type;
      return 2;
    }

    Ptr<detail::RotationWarper> warper = warper_creator->create(
        static_cast<float>(warped_image_scale * seam_work_aspect));
    for (int i = 0; i < images_n; ++i) {
      Mat_<float> K;
      cameras[i].K().convertTo(K, CV_32F);
      float swa = static_cast<float>(seam_work_aspect);
      K(0,0) *= swa; K(0,2) *= swa;  // NOLINT
      K(1,1) *= swa; K(1,2) *= swa;  // NOLINT

      corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
      sizes[i] = images_warped[i].size();

      warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    for (int i = 0; i < images_n; ++i)
      images_warped[i].convertTo(images_warped_f[i], CV_32F);

    logger->AddSplit("warp images");
  }
  logger->DumpToLog();
  LOG(INFO);
  applog->AddSplit("images warping");

  // exposure compensation
  logger->Reset("Exposure Compensation");
  Ptr<detail::ExposureCompensator> compensator;
  {
    int type;
    if (expos_comp_type == "no")                    type = detail::ExposureCompensator::NO;
    else if (expos_comp_type == "gain")             type = detail::ExposureCompensator::GAIN;
    else if (expos_comp_type == "gain_blocks")      type = detail::ExposureCompensator::GAIN_BLOCKS;
    else if (expos_comp_type == "channels")         type = detail::ExposureCompensator::CHANNELS;
    else if (expos_comp_type == "channels_blocks")  type = detail::ExposureCompensator::CHANNELS_BLOCKS;
    else {  // NOLINT
      LOG(FATAL) << "Bad exposure compensation method";
      return 2;
    }

    compensator = detail::ExposureCompensator::createDefault(type);

    if (dynamic_cast<detail::GainCompensator*>(compensator.get())) {
      detail::GainCompensator *gcompensator = dynamic_cast<detail::GainCompensator*>(compensator.get());
      gcompensator->setNrFeeds(expos_comp_nr_feeds);
    }
    if (dynamic_cast<detail::ChannelsCompensator*>(compensator.get())) {
      detail::ChannelsCompensator *ccompensator = dynamic_cast<detail::ChannelsCompensator*>(compensator.get());
      ccompensator->setNrFeeds(expos_comp_nr_feeds);
    }
    if (dynamic_cast<detail::BlocksCompensator*>(compensator.get())) {
      detail::BlocksCompensator *bcompensator = dynamic_cast<detail::BlocksCompensator*>(compensator.get());
      bcompensator->setNrFeeds(expos_comp_nr_feeds);
      bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
      bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
    }

    compensator->feed(corners, images_warped, masks_warped);
    logger->AddSplit("feed");
  }
  logger->DumpToLog();
  LOG(INFO);
  applog->AddSplit("exposure compensation");

  // seam estimation
  logger->Reset("Seam Estimation");
  {
    Ptr<detail::SeamFinder> seam_finder;
    if (seam_find_type == "no") {
      seam_finder = makePtr<detail::NoSeamFinder>();
    } else if (seam_find_type == "voronoi") {
      seam_finder = makePtr<detail::VoronoiSeamFinder>();
    } else if (seam_find_type == "gc_color") {
// #ifdef HAVE_OPENCV_CUDALEGACY
//       if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
//         seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(detail::GraphCutSeamFinderBase::COST_COLOR);
//       else
// #endif
      seam_finder = makePtr<detail::GraphCutSeamFinder>(detail::GraphCutSeamFinderBase::COST_COLOR);
    } else if (seam_find_type == "gc_colorgrad") {
// #ifdef HAVE_OPENCV_CUDALEGACY
//       if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
//         seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
//       else
// #endif
      seam_finder = makePtr<detail::GraphCutSeamFinder>(detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
    } else if (seam_find_type == "dp_color") {
      seam_finder = makePtr<detail::DpSeamFinder>(detail::DpSeamFinder::COLOR);
    } else if (seam_find_type == "dp_colorgrad") {
      seam_finder = makePtr<detail::DpSeamFinder>(detail::DpSeamFinder::COLOR_GRAD);
    }
    if (!seam_finder) {
      LOG(FATAL) << "Can't create the following seam finder '" << seam_find_type;
      return 2;
    }

    seam_finder->find(images_warped_f, corners, masks_warped);
    logger->AddSplit("find");
  }
  logger->DumpToLog();
  LOG(INFO);
  applog->AddSplit("seam estimation");

  // release unused memory
  images.clear();
  images_warped.clear();
  images_warped_f.clear();
  masks.clear();

  // image blenders
  logger->Reset("Image Blenders");
  {
    int blend_type_;
    if (blend_type == "no")             blend_type_ = detail::Blender::NO;
    else if (blend_type == "feather")   blend_type_ = detail::Blender::FEATHER;
    else if (blend_type == "multiband") blend_type_ = detail::Blender::MULTI_BAND;
    else {  // NOLINT
      LOG(FATAL) << "Bad blending method";
      return 2;
    }

    bool timelapse = true;
    int timelapse_type_ = 0;
    if (timelapse_type == "no")         timelapse = false;
    else if (timelapse_type == "as_is") timelapse_type = detail::Timelapser::AS_IS;
    else if (timelapse_type == "crop")  timelapse_type = detail::Timelapser::CROP;
    else {  // NOLINT
      LOG(FATAL) << "Bad timelapse method";
      return 2;
    }

    double compose_scale = 1;
    bool is_compose_scale_set = false;

    Ptr<detail::RotationWarper> warper;

    Mat img;
    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<detail::Blender> blender;
    Ptr<detail::Timelapser> timelapser;
    // double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < images_n; ++img_idx) {
      stringstream ss;
      ss << "image #" << indices[img_idx]+1;
      auto img_name = ss.str();
      LOG(INFO) << "Compositing " << img_name;

      // read image and resize it if necessary
      // auto img_full = imread(samples::findFile(images_name[img_idx]));
      auto img_full = images_full[img_idx];

      if (!is_compose_scale_set) {
        if (compose_megapix > 0)
          compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / img_full.size().area()));
        is_compose_scale_set = true;

        // compute relative scales
        // compose_seam_aspect = compose_scale / seam_scale;
        compose_work_aspect = compose_scale / work_scale;

        // update warped image scale
        warped_image_scale *= static_cast<float>(compose_work_aspect);
        warper = warper_creator->create(warped_image_scale);

        // update corners and sizes
        for (int i = 0; i < images_n; ++i) {
          // update intrinsics
          cameras[i].focal *= compose_work_aspect;
          cameras[i].ppx *= compose_work_aspect;
          cameras[i].ppy *= compose_work_aspect;

          // update corner and size
          Size sz = images_full_size[i];
          if (abs(compose_scale - 1) > 1e-1) {
            sz.width = cvRound(images_full_size[i].width * compose_scale);
            sz.height = cvRound(images_full_size[i].height * compose_scale);
          }

          Mat K;
          cameras[i].K().convertTo(K, CV_32F);
          Rect roi = warper->warpRoi(sz, K, cameras[i].R);
          corners[i] = roi.tl();
          sizes[i] = roi.size();
        }
      }

      if (abs(compose_scale - 1) > 1e-1)
        resize(img_full, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
      else
        img = img_full;
      img_full.release();

      // logger->AddSplit(img_name + ", read");

      Size img_size = img.size();

      Mat K;
      cameras[img_idx].K().convertTo(K, CV_32F);

      // warp the current image
      warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

      // warp the current image mask
      mask.create(img_size, CV_8U);
      mask.setTo(Scalar::all(255));
      warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

      // compensate exposure
      compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

      img_warped.convertTo(img_warped_s, CV_16S);
      img_warped.release();
      img.release();
      mask.release();

      dilate(masks_warped[img_idx], dilated_mask, Mat());
      resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
      mask_warped = seam_mask & mask_warped;

      if (!blender && !timelapse) {
        blender = detail::Blender::createDefault(blend_type_, try_cuda);
        Size dst_sz = detail::resultRoi(corners, sizes).size();
        float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
        if (blend_width < 1.f) {
          blender = detail::Blender::createDefault(detail::Blender::NO, try_cuda);
        } else if (blend_type_ == detail::Blender::MULTI_BAND) {
          detail::MultiBandBlender* mb = dynamic_cast<detail::MultiBandBlender*>(blender.get());
          mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
          LOG(INFO) << "Multi-band blender, number of bands: " << mb->numBands();
        } else if (blend_type_ == detail::Blender::FEATHER) {
          detail::FeatherBlender* fb = dynamic_cast<detail::FeatherBlender*>(blender.get());
          fb->setSharpness(1.f/blend_width);
          LOG(INFO) << "Feather blender, sharpness: " << fb->sharpness();
        }
        blender->prepare(corners, sizes);
      } else if (!timelapser && timelapse) {
        timelapser = detail::Timelapser::createDefault(timelapse_type_);
        timelapser->initialize(corners, sizes);
      }

      // blend the current image
      if (timelapse) {
        timelapser->process(img_warped_s, Mat::ones(img_warped_s.size(), CV_8UC1), corners[img_idx]);
        String fixedFileName;
        size_t pos_s = String(images_name[img_idx]).find_last_of("/\\");
        if (pos_s == String::npos) {
          fixedFileName = "fixed_" + images_name[img_idx];
        } else {
          fixedFileName = "fixed_" + String(images_name[img_idx]).substr(pos_s + 1, String(images_name[img_idx]).length() - pos_s);
        }
        logger->AddSplit(img_name + ", timelapser process");
        imwrite(fixedFileName, timelapser->getDst());
        LOG(INFO) << "Output timelapser result to " << fixedFileName;
      } else {
        blender->feed(img_warped_s, mask_warped, corners[img_idx]);
        logger->AddSplit(img_name + ", blender feed");
      }
    }

    if (!timelapse) {
      Mat result, result_mask;
      blender->blend(result, result_mask);
      logger->AddSplit("blender blend");
      imwrite(result_name, result);
      LOG(INFO) << "Output blender result to " << result_name;
    }
  }
  logger->DumpToLog();
  LOG(INFO);
  applog->AddSplit("image blenders");

  applog->DumpToLog();
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
