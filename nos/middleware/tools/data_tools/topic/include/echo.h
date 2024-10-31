#pragma once
#include <memory>
#include <mutex>
#include <queue>

namespace hozon {
namespace netaos {
namespace topic {

class EchoImpl;

struct EchoOptions {
    bool show_help_info = false;
    std::vector<std::string> topics;
    bool open_proto_log = false;
    std::string json_format_path = "./";
};

class Echo {
   public:
    Echo();
    void Start(EchoOptions echo_options);
    void Stop();
    ~Echo();

   private:
    std::unique_ptr<EchoImpl> echo_impl_;
};

}  // namespace topic
}  //namespace netaos
}  //namespace hozon