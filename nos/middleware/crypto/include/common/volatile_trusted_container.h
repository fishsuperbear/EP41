#ifndef ARA_CRYPTO_COMMON_VOLATILE_TRUSTED_CONTAINER_H_
#define ARA_CRYPTO_COMMON_VOLATILE_TRUSTED_CONTAINER_H_
#include "io_interface.h"

namespace hozon {
namespace netaos {
namespace crypto {

class VolatileTrustedContainer {
public:
    using Uptr = std::unique_ptr<VolatileTrustedContainer>;
    virtual ~VolatileTrustedContainer() noexcept = default;
    virtual IOInterface& GetIOInterface () const noexcept=0;
    VolatileTrustedContainer& operator=(const VolatileTrustedContainer& other) = default;
    VolatileTrustedContainer& operator=(VolatileTrustedContainer&& other) = default;
};

}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_COMMON_VOLATILE_TRUSTED_CONTAINER_H_