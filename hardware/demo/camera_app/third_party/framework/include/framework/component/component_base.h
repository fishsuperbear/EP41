#ifndef CYBER_COMPONENT_COMPONENT_BASE_H_
#define CYBER_COMPONENT_COMPONENT_BASE_H_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "gflags/gflags.h"

#include "framework/proto/component_conf.pb.h"

#include "framework/class_loader/class_loader.h"
#include "framework/common/environment.h"
#include "framework/common/file.h"
#include "framework/node/node.h"
#include "framework/scheduler/scheduler.h"

namespace netaos {
namespace framework {

using netaos::framework::proto::ComponentConfig;
using netaos::framework::proto::TimerComponentConfig;

class ComponentBase : public std::enable_shared_from_this<ComponentBase> {
 public:
  template <typename M>
  using Reader = framework::Reader<M>;

  virtual ~ComponentBase() {}

  virtual bool Initialize(const ComponentConfig& config) { return false; }
  virtual bool Initialize(const TimerComponentConfig& config) { return false; }
  virtual void Shutdown() {
    if (is_shutdown_.exchange(true)) {
      return;
    }

    Clear();
    for (auto& reader : readers_) {
      reader->Shutdown();
    }
    scheduler::Instance()->RemoveTask(node_->Name());
  }

  template <typename T>
  bool GetProtoConfig(T* config) const {
    return common::GetProtoFromFile(config_file_path_, config);
  }

 protected:
  virtual bool Init() = 0;
  virtual void Clear() { return; }
  const std::string& ConfigFilePath() const { return config_file_path_; }

  void LoadConfigFiles(const ComponentConfig& config) {
    if (!config.config_file_path().empty()) {
      if (config.config_file_path()[0] != '/') {
        config_file_path_ = common::GetAbsolutePath(common::WorkRoot(),
                                                    config.config_file_path());
      } else {
        config_file_path_ = config.config_file_path();
      }
    }

    if (!config.flag_file_path().empty()) {
      std::string flag_file_path = config.flag_file_path();
      if (flag_file_path[0] != '/') {
        flag_file_path =
            common::GetAbsolutePath(common::WorkRoot(), flag_file_path);
      }
      google::SetCommandLineOption("flagfile", flag_file_path.c_str());
    }
  }

  void LoadConfigFiles(const TimerComponentConfig& config) {
    if (!config.config_file_path().empty()) {
      if (config.config_file_path()[0] != '/') {
        config_file_path_ = common::GetAbsolutePath(common::WorkRoot(),
                                                    config.config_file_path());
      } else {
        config_file_path_ = config.config_file_path();
      }
    }

    if (!config.flag_file_path().empty()) {
      std::string flag_file_path = config.flag_file_path();
      if (flag_file_path[0] != '/') {
        flag_file_path =
            common::GetAbsolutePath(common::WorkRoot(), flag_file_path);
      }
      google::SetCommandLineOption("flagfile", flag_file_path.c_str());
    }
  }

  std::atomic<bool> is_shutdown_ = {false};
  std::shared_ptr<Node> node_ = nullptr;
  std::string config_file_path_ = "";
  std::vector<std::shared_ptr<ReaderBase>> readers_;
};

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_COMPONENT_COMPONENT_BASE_H_
