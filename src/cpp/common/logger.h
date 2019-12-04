#pragma once

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// log config

// #define LOG_PREFIX
#ifndef LOG_STREAM
#  define LOG_STREAM std::cout
#endif
#ifndef LOG_MINLEVEL
#  define LOG_MINLEVEL Logger::INFO
#endif
#ifndef LOG_V
#  define LOG_V 0
#endif

#define LOG_TIMING
#ifdef LOG_TIMING_OFF
#  undef LOG_TIMING
#endif
#ifndef LOG_TIMING_STREAM
#  define LOG_TIMING_STREAM std::cout
#endif

// log macros

#define LOG_IF(severity, condition) !(condition) ? (void)0 : LoggerVoidify() & \
    Logger((char *)__FILE__, __LINE__, "native", Logger::severity).stream()  // NOLINT

#define LOG(severity) LOG_IF(severity, Logger::severity >= LOG_MINLEVEL)

// VLOG macros always log at the INFO log level
#define VLOG(n)       LOG_IF(INFO, Logger::INFO >= LOG_MINLEVEL && n <= LOG_V)
#define VLOG_IS_ON(n) (n <= LOG_V)

// other macros

#if !defined(OS_PATH_SEP)
#  if defined(_WIN32) || defined(__CYGWIN__)
#    define OS_PATH_SEP '\\'
#  else
#    define OS_PATH_SEP '/'
#  endif
#endif

// Logger

class Logger {
 public:
  enum Severity {
    INFO      = 0,
    WARNING   = 1,
    ERROR     = 2,
    FATAL     = 3,
  };

  Logger(const char *file, int line, const char *tag, int severity,
      std::ostream &os = LOG_STREAM)
    : file_(file), line_(line), tag_(tag), severity_(severity), ostream_(os) {
#ifdef LOG_PREFIX
    StripBasename(std::string(file), &filename_only_);
    stream_ << SeverityLabel() << "/" << filename_only_ << ":" << line_ << " ";
#else
    (void)line_;
#endif
  }

  ~Logger() noexcept(false) {
    stream_ << std::endl;
    ostream_ << stream_.str();
    if (severity_ == FATAL) {
      abort();
    }
  }

  std::stringstream &stream() { return stream_; }

  template<class T>
  Logger &operator<<(const T &val) {
    stream_ << val;
    return *this;
  }

  template<class T>
  Logger &operator<<(T &&val) {
    stream_ << std::move(val);
    return *this;
  }

 private:
  void StripBasename(const std::string &full_path, std::string *filename) {
    auto pos = full_path.rfind(OS_PATH_SEP);
    if (pos != std::string::npos) {
      *filename = full_path.substr(pos + 1, std::string::npos);
    } else {
      *filename = full_path;
    }
  }

  char SeverityLabel() {
    switch (severity_) {
      case INFO:      return 'I';
      case WARNING:   return 'W';
      case ERROR:     return 'E';
      case FATAL:     return 'F';
      default:        return 'V';
    }
  }

  std::stringstream stream_;

  std::string file_;
  std::string filename_only_;
  int line_;
  std::string tag_;
  int severity_;

  std::ostream &ostream_;
};

class LoggerVoidify {
 public:
  LoggerVoidify() = default;
  void operator&(const std::ostream &/*s*/) {}
};

// TimingLogger

class TimingLogger {
 public:
  TimingLogger() = default;
  virtual ~TimingLogger() = default;
  static std::shared_ptr<TimingLogger> Create(std::string /*label*/);
  virtual void Reset(std::string /*label*/) {}
  virtual void Reset() {}
  virtual void AddSplit(std::string /*split_label*/) {}
  virtual void DumpToLog(std::ostream &/*os*/ = LOG_TIMING_STREAM) {}
};

class TimingLoggerImpl : public TimingLogger {
 public:
  using clock = std::chrono::system_clock;

  explicit TimingLoggerImpl(std::string label) {
    Reset(std::move(label));
  }

  void Reset(std::string label) override {
    label_ = std::move(label);
    Reset();
  }

  void Reset() override {
    split_labels_.clear();
    split_times_.clear();
    AddSplit("");
  }

  void AddSplit(std::string split_label) override {
    split_labels_.push_back(std::move(split_label));
    split_times_.push_back(clock::now());
  }

  void DumpToLog(std::ostream &os = LOG_TIMING_STREAM) override {
    using namespace std::chrono;
    os << label_ << ": begin" << std::endl;
    auto first = split_times_[0];
    auto now = first;
    for (std::size_t i = 1, n = split_times_.size(); i < n; i++) {
      now = split_times_[i];
      auto split_label = split_labels_[i];
      auto prev = split_times_[i - 1];
      os << label_ << ":      "
         << duration_cast<milliseconds>(now - prev).count() << " ms, "
         << split_label << std::endl;
    }
    os << label_ << ": end, "
       << duration_cast<milliseconds>(now - first).count() << " ms"
       << std::endl;
  }

 private:
  std::string label_;
  std::vector<std::string> split_labels_;
  std::vector<clock::time_point> split_times_;
};

inline std::shared_ptr<TimingLogger> TimingLogger::Create(std::string label) {
#ifdef LOG_TIMING
  return std::make_shared<TimingLoggerImpl>(std::move(label));
#else
  (void)label;
  return std::make_shared<TimingLogger>();
#endif
}
