#include "common/camera.h"

// https://stackoverflow.com/questions/10167534/
// how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
std::string type2string(int type);

int main(int argc, char const *argv[]) {
  int index = 0;
  if (argc >= 2) {
    try {
      index = std::stoi(argv[1]);
    } catch (...) {
      std::cout << "Warning: Can't set index: " << argv[1] << std::endl;
    }
  }
  Camera cam(index);
  if (!cam.IsOpened()) {
    std::cerr << "Error: Open camera failed" << std::endl;
    return 1;
  }
  std::cout << "\033[1;32mPress ESC/Q to terminate\033[0m\n";

  // std::cout << cv::getBuildInformation();

  // cv::VideoCaptureProperties:
  // http://docs.opencv.org/master/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
  // For get raw camera data with v4l2
  /*
  std::cout << "CAP_PROP_CONVERT_RGB disabled: "
        << cam.Set(cv::CAP_PROP_CONVERT_RGB, 0) << std::endl;
  std::cout << "CAP_PROP_FORMAT: " << cam.Get(cv::CAP_PROP_FORMAT) << std::endl;
  std::cout << std::flush;
  */

  cam.Capture([&cam](const cv::Mat &frame, const cv::Mat &/*depthmap*/) {
    // std::cout << "type: " << type2string(frame.type()) << std::endl;
    // cv::cvtColor(frame, frame, cv::COLOR_YUV2BGR_YUY2);
    cam.DrawInfo(frame);
    cv::imshow("frame", frame);

    char key = static_cast<char>(cv::waitKey(10));
    return !(key == 27 || key == 'q' || key == 'Q');  // ESC/Q
  });

  return 0;
}

std::string type2string(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
