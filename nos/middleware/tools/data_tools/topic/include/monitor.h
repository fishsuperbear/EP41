
#pragma once
#include <iostream>
#include <memory>
#include <vector>

namespace hozon {
namespace netaos {
namespace topic {

class MonitorImpl;

struct MonitorOptions {
    bool show_help_info = false;
    bool monitor_all = false;
    std::vector<std::string> events;
    bool open_proto_log = false;
    bool method = false;
};

class Monitor {
   public:
    Monitor();
    ~Monitor();
    void Start(MonitorOptions monitor_options);
    void Stop();
    void SigResizeHandle();

   private:
    std::unique_ptr<MonitorImpl> monitor_impl_;
};

}  // namespace topic
}  //namespace netaos
}  //namespace hozon