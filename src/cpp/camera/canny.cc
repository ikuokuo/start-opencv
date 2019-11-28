#include "common/camera.h"

#include <opencv2/imgcodecs.hpp>

int main(int /*argc*/, char const */*argv*/[]) {
  Camera cam(0);
  if (!cam.IsOpened()) {
    std::cerr << "ERROR: Open camera failed" << std::endl;
    return 1;
  }
  std::cout << "\033[1;32mPress ESC/Q to terminate\033[0m\n\n";

  cv::Mat edges;
  cam.Capture([&cam, &edges](const cv::Mat &frame,
      const cv::Mat &/*depthmap*/) {
    cv::cvtColor(frame, edges, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(edges, edges, cv::Size(7, 7), 1.5, 1.5);
    cv::Canny(edges, edges, 0, 30, 3);
    cam.DrawInfo(edges);
    cv::imshow("edges", edges);

    char key = static_cast<char>(cv::waitKey(10));
    return !(key == 27 || key == 'q' || key == 'Q');  // ESC/Q
  });

  return 0;
}
