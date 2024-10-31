#ifndef ARA_CRYPTO_CRYP_IMP_VOLATILE_TRUSTED_CONTAINE_H_
#define ARA_CRYPTO_CRYP_IMP_VOLATILE_TRUSTED_CONTAINE_H_

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <utility>

#include "common/volatile_trusted_container.h"
#include "common/imp_io_interface.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class ImpVolatileTrustedContainer:public VolatileTrustedContainer {
public:
    ImpVolatileTrustedContainer(std::size_t capacity):capacity_(capacity){
        pio_ = new ImpIOInterface(capacity_);
    }
    ImpVolatileTrustedContainer() = default;
    IOInterface& GetIOInterface () const noexcept override{
        return *pio_;
        // auto uptr = std::make_unique<ImpIOInterface>(capacity_);
        // return *uptr.get();
        // return const_cast<ImpIOInterface&>(io);
    };

private:
    std::size_t capacity_ = 0;
    ImpIOInterface *pio_;
};

}  // namespace cryp
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon

#endif  // #define ARA_CRYPTO_CRYP_IMP_VOLATILE_TRUSTED_CONTAINE_H_