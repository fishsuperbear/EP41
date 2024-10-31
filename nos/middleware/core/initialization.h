#ifndef NETAOS_CORE_INITIALIZATION_H
#define NETAOS_CORE_INITIALIZATION_H

#include "core/result.h"
namespace hozon {

namespace netaos {
namespace core {
Result<void> Initialize();
Result<void> Deinitialize();
}  // End of namespace core
}  // End of namespace netaos
}  // namespace hozon
#endif
