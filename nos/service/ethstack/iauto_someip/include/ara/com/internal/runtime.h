#ifndef ARA_COM_INTERNAL_RUNTIME_H_
#define ARA_COM_INTERNAL_RUNTIME_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <ara/core/instance_specifier.h>
#include <ara/core/string.h>
#include <ara/core/vector.h>
#include <algorithm>

#include "ara/com/types.h"

namespace ara {
namespace com {
namespace runtime {

/**
 * @brief Translate an InstanceSpecifier to a Instance Identifiers list
 *
 * @uptrace{SWS_CM_00118}
 *
 * @param modelName
 * @return ara::com::InstanceIdentifierContainer
 */
extern ara::com::InstanceIdentifierContainer ResolveInstanceIDs(ara::core::InstanceSpecifier modelName);


}  // namespace runtime
}  // namespace com
}  // namespace ara

#endif  // ARA_COM_INTERNAL_RUNTIME_H_
/* EOF */
