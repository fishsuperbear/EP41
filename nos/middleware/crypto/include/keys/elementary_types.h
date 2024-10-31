#ifndef ARA_CRYPTO_KEYS_ELEMENTARY_TYPES_H_
#define ARA_CRYPTO_KEYS_ELEMENTARY_TYPES_H_
#include "core/vector.h"


namespace hozon {
namespace netaos {
namespace crypto {
namespace keys {

class KeySlot;

using TransactionId = std::uint64_t;
using TransactionScope = netaos::core::Vector<KeySlot*>;



}  // namespace KEYS
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_KEYS_ELEMENTARY_TYPES_H_