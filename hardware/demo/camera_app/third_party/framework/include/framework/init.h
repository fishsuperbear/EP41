#ifndef CYBER_INIT_H_
#define CYBER_INIT_H_

#include "framework/common/log.h"
#include "framework/state.h"

namespace netaos {
namespace framework {

bool Init(const char* binary_name);
void Clear();

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_INIT_H_
