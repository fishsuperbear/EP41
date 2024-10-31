#pragma once
#include <iostream>
#include <memory>
#include <vector>

namespace hozon {
namespace netaos {
namespace topic {
class HzImpl;

struct HzOptions {
    bool show_help_info = false;
    uint skip_sample_num = 5;
    bool monitor_all = false;
    bool method = false;
    std::vector<std::string> events;
    //Unit s. defult 0.
    uint window_duration = 0;
    //设置自动退出时间，单位秒
    int exit_time = 0;
};

class Hz {
   public:
    Hz();
    void Start(HzOptions hz_options);
    void Stop();
    ~Hz();

   private:
    std::unique_ptr<HzImpl> hz_impl_;
};

}  // namespace topic
}  //namespace netaos
}  //namespace hozon