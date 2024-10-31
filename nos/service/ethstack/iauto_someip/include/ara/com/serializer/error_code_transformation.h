#ifndef ERROR_CODE_TRANSFORMATION_H_
#define ERROR_CODE_TRANSFORMATION_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <stdint.h>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <ara/core/optional.h>
#include <ara/core/error_code.h>
#include <ara/core/error_domain.h>

namespace ara {
namespace com {
namespace runtime {

extern int32_t serializeErrorCode(const ara::core::ErrorCode& error_code, std::vector<uint8_t>& data);
extern ara::core::ErrorCode deserializeErrorCode(const std::vector<uint8_t>& data);

}  // namespace runtime
}  // namespace com
}  // namespace ara

#endif  // ERROR_CODE_TRANSFORMATION_H_
/* EOF */
