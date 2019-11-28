#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

class Camera {
 public:
  using VideoCapture = cv::VideoCapture;
  using FrameCallback = std::function<bool(const cv::Mat &, const cv::Mat &)>;

  explicit Camera(int index = 0)
    : cap_(new VideoCapture(std::move(index))),
#ifdef WITH_OPENNI
      use_openni_(index == cv::CAP_OPENNI || index == cv::CAP_OPENNI2),
#endif
      fps_(-1) {
    int flag = 0;
#ifdef WITH_OPENNI
    if (use_openni_) {
      flag = cv::CAP_OPENNI_IMAGE_GENERATOR;
      Set(cv::CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, cv::CAP_OPENNI_VGA_30HZ);
    }
#endif
    std::cout << "Capture frame width: " << Get(flag + cv::CAP_PROP_FRAME_WIDTH)
      << ", height: " << Get(flag + cv::CAP_PROP_FRAME_HEIGHT)
      << std::endl;
    fps_ = Get(flag + cv::CAP_PROP_FPS);
    std::cout << "Capture fps: " << fps_ << std::endl
#ifdef WITH_OPENNI
      << "Use OpenNI: " << (use_openni_ ? "true" : "false") << std::endl
#endif
      << std::endl;
  }
  virtual ~Camera() {}

  bool UseOpenNI() {
#ifdef WITH_OPENNI
    return use_openni_;
#else
    return false;
#endif
  }

  double Get(int prop_id) const {
    return cap_->get(prop_id);
  }

  bool Set(int prop_id, double value) {
    return cap_->set(prop_id, value);
  }

  bool IsOpened() {
#ifdef WITH_OPENNI
    return use_openni_ || cap_->isOpened();
#else
    return cap_->isOpened();
#endif
  }

  void Preview(FrameCallback callback = nullptr) {
    cv::namedWindow("frame");
    Capture([&callback](const cv::Mat &frame, const cv::Mat &depthmap) {
      cv::imshow("frame", frame);
      if (callback && !callback(frame, depthmap)) {
        // return false to break
        return false;
      }
      // return false to break if ESC/Q
      char key = static_cast<char>(cv::waitKey(30));
      return !(key == 27 || key == 'q' || key == 'Q');
    });
  }

  // callback return true to continue and false to break
  void Capture(FrameCallback callback) {
    if (!callback) {
      std::cerr << "ERROR: Null FrameCallback" << std::endl;
      return;
    }

    cv::Mat frame;
    cv::Mat depthmap;
    int64 t;
    for (;;) {
      t = cv::getTickCount();

#ifdef WITH_OPENNI
      if (use_openni_) {
        cap_->grab();

        cap_->retrieve(depthmap, cv::CAP_OPENNI_DEPTH_MAP);
        cap_->retrieve(frame, cv::CAP_OPENNI_BGR_IMAGE);
        // cap_->retrieve(frame, cv::CAP_OPENNI_GRAY_IMAGE);
      } else {
        cap_->read(frame);
      }
#else
      cap_->read(frame);
#endif

      if (frame.empty()) {
        std::cerr << "ERROR: Blank frame grabbed" << std::endl;
        break;
      }

      bool ok = callback(frame, depthmap);

      fps_ = cv::getTickFrequency() / (cv::getTickCount() - t);
      // std::cout << "fps: " << fps_ << std::endl;

      if (!ok) break;
    }
  }

  double FPS() const {
    return fps_;
  }

  std::string ExtraInfo() const {
    std::ostringstream info;
    info << "FPS: " << fps_;
    return info.str();
  }

  void DrawInfo(const cv::Mat &im) const {
    using namespace std;  // NOLINT
    int w = im.cols, h = im.rows;
    // topLeft: width x height
    {
      ostringstream ss;
      ss << w << "x" << h;
      string text = ss.str();

      int baseline = 0;
      cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_PLAIN,
        1, 1, &baseline);

      cv::putText(im, text, cv::Point(5, 5 + textSize.height),
          cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 255));
    }
    // topRight: fps
    {
      ostringstream ss;
      ss << "FPS: " << fps_;
      string text = ss.str();

      int baseline = 0;
      cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_PLAIN,
        1, 1, &baseline);

      cv::putText(im, text,
          cv::Point(w - 5 - textSize.width, 5 + textSize.height),
          cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 255));
    }
  }

 private:
  std::unique_ptr<VideoCapture> cap_;
#ifdef WITH_OPENNI
  bool use_openni_;
#endif
  double fps_;
};
