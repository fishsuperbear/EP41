#ifndef CYBER_CYBER_H_
#define CYBER_CYBER_H_

#include <memory>
#include <string>
#include <utility>

#include "framework/common/log.h"
#include "framework/component/component.h"
#include "framework/init.h"
#include "framework/node/node.h"
#include "framework/task/task.h"
#include "framework/time/time.h"
#include "framework/timer/timer.h"

namespace netaos {
namespace framework {

std::unique_ptr<Node> CreateNode(const std::string& node_name,
                                 const std::string& name_space = "");

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_CYBER_H_
