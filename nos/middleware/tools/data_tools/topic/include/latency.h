#pragma once
#include <iostream>
#include <memory>
#include <vector>

namespace hozon {
namespace netaos {
namespace topic {
class LatencyImpl;

struct LatencyOptions {
    bool show_help_info = false;
    std::vector<std::string> topics;
};

class Latency {
   public:
    Latency();
    void Start(LatencyOptions latency_options);
    void Stop();
    ~Latency();

   private:
    std::unique_ptr<LatencyImpl> latency_impl_;
};

}  // namespace topic
}  //namespace netaos
}  //namespace hozon