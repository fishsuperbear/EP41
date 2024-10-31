#ifndef ARA_CRYPTO_COMMON_MEM_REGION_H_
#define ARA_CRYPTO_COMMON_MEM_REGION_H_
#include <cstdint>
#include "core/span.h"

namespace hozon {
namespace netaos {
namespace crypto {
using ReadOnlyMemRegion = netaos::core::Span<const std::uint8_t>;
using ReadWriteMemRegion = netaos::core::Span<std::uint8_t>;

}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_COMMON_MEM_REGION_H_