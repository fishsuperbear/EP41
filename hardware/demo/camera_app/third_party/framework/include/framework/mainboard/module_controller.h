#ifndef CYBER_MAINBOARD_MODULE_CONTROLLER_H_
#define CYBER_MAINBOARD_MODULE_CONTROLLER_H_

#include <memory>
#include <string>
#include <vector>

#include "framework/proto/dag_conf.pb.h"

#include "framework/class_loader/class_loader_manager.h"
#include "framework/component/component.h"
#include "framework/mainboard/module_argument.h"

namespace netaos {
namespace framework {
namespace mainboard {

using netaos::framework::proto::DagConfig;

class ModuleController {
 public:
  explicit ModuleController(const ModuleArgument& args);
  virtual ~ModuleController() = default;

  bool Init();
  bool LoadAll();
  void Clear();

 private:
  bool LoadModule(const std::string& path);
  bool LoadModule(const DagConfig& dag_config);
  int GetComponentNum(const std::string& path);
  int total_component_nums = 0;
  bool has_timer_component = false;

  ModuleArgument args_;
  class_loader::ClassLoaderManager class_loader_manager_;
  std::vector<std::shared_ptr<ComponentBase>> component_list_;
};

inline ModuleController::ModuleController(const ModuleArgument& args)
    : args_(args) {}

inline bool ModuleController::Init() { return LoadAll(); }

}  // namespace mainboard
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_MAINBOARD_MODULE_CONTROLLER_H_
