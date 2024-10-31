#ifndef ARA_CRYPTO_CRYP_IMP_CRYPTO_SERVICE_H_
#define ARA_CRYPTO_CRYP_IMP_CRYPTO_SERVICE_H_

#include "common/base_id_types.h"
#include "common/crypto_object_uid.h"
#include "cryp/extension_service.h"
#include "cryp/crypto_service.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class ImpCryptoService : public CryptoService{
public:
    using Uptr = std::unique_ptr<ImpCryptoService>;
    std::size_t GetBlockSize() const noexcept override;
    std::size_t GetMaxInputSize(bool suppressPadding = false) const noexcept override;
    std::size_t GetMaxOutputSize(bool suppressPadding = false) const noexcept override;
    std::size_t GetActualKeyBitLength () const noexcept override;
    CryptoObjectUid GetActualKeyCOUID () const noexcept override;
    AllowedUsageFlags GetAllowedUsage () const noexcept override;
    std::size_t GetMaxKeyBitLength () const noexcept override;
    std::size_t GetMinKeyBitLength () const noexcept override;
    bool IsKeyBitLengthSupported(std::size_t keyBitLength) const noexcept override;
    bool IsKeyAvailable () const noexcept override;
private:
    std::size_t  blockSize_;
    std::size_t  maxInputSize_;
    std::size_t  maOutputSize_;
    std::size_t  actualKeyBitLength_;
    std::size_t  maxKeyBitLength_;
    std::size_t  minKeyBitLength_;
    CryptoObjectUid cryObjUid_;
    AllowedUsageFlags allowedUsageFlags_;
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_IMP_CRYPTO_SERVICE_H_