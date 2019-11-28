#include "common/camera.h"

int main(int /*argc*/, char const */*argv*/[]) {
  Camera cam(cv::CAP_OPENNI2);
  if (!cam.IsOpened()) {
    std::cerr << "ERROR: Open camera failed" << std::endl;
    return 1;
  }
  std::cout << "\033[1;32mPress ESC/Q to terminate\033[0m\n\n";

  double min, max;
  cv::Mat adjmap, colormap;
  cam.Capture([&](const cv::Mat &frame, const cv::Mat &depthmap) {
    cv::minMaxIdx(depthmap, &min, &max);
    cv::convertScaleAbs(depthmap, adjmap, 255 / max);

    // cv::applyColorMap(adjmap, colormap, cv::COLORMAP_AUTUMN);

    cam.DrawInfo(frame);
    cam.DrawInfo(adjmap);
    cv::imshow("frame", frame);
    cv::imshow("depthmap", adjmap);

    char key = static_cast<char>(cv::waitKey(10));
    return !(key == 27 || key == 'q' || key == 'Q');  // ESC/Q
  });

  return 0;
}

// OpenCV: How to visualize a depth image:
//   http://stackoverflow.com/questions/13840013/opencv-how-to-visualize-a-depth-image
