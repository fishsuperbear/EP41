#ifndef CYBER_SCHEDULER_SCHEDULER_FACTORY_H_
#define CYBER_SCHEDULER_SCHEDULER_FACTORY_H_

#include "framework/common/environment.h"
#include "framework/common/file.h"
#include "framework/common/global_data.h"
#include "framework/common/util.h"
#include "framework/scheduler/policy/scheduler_choreography.h"
#include "framework/scheduler/policy/scheduler_classic.h"
#include "framework/scheduler/scheduler.h"

namespace netaos {
namespace framework {
namespace scheduler {

Scheduler* Instance();
void CleanUp();

}  // namespace scheduler
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SCHEDULER_SCHEDULER_FACTORY_H_
