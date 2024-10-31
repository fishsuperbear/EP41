#ifndef ARA_CRYPTO_CRYP_DIGEST_SERVICE_H_
#define ARA_CRYPTO_CRYP_DIGEST_SERVICE_H_
#include <memory>
#include "core/result.h"
#include "common/mem_region.h"
#include "cryp/block_service.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class DigestService:public BlockService {
public:
    using Uptr = std::unique_ptr<DigestService>;
    virtual netaos::core::Result<bool> Compare(ReadOnlyMemRegion expected, std::size_t offset = 0) const noexcept = 0;
    virtual std::size_t GetDigestSize() const noexcept = 0;
    virtual bool IsFinished() const noexcept = 0;
    virtual bool IsStarted() const noexcept = 0;

   private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_DIGEST_SERVICE_H_