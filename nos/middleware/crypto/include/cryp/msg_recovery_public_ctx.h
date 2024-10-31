#ifndef ARA_CRYPTO_CRYP_MSG_RECOVERY_PUBLIC_CTX_H_
#define ARA_CRYPTO_CRYP_MSG_RECOVERY_PUBLIC_CTX_H_
#include "core/result.h"
#include "cryp/crypto_context.h"
#include "cryp/cryobj/public_key.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class MsgRecoveryPublicCtx:public CryptoContext{
public:

    using Uptr = std::unique_ptr<MsgRecoveryPublicCtx>;
    virtual ExtensionService::Uptr GetExtensionService() const noexcept = 0;
    virtual std::size_t GetMaxInputSize(bool suppressPadding = false) const noexcept = 0;
    virtual std::size_t GetMaxOutputSize(bool suppressPadding = false) const noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > DecodeAndVerify(ReadOnlyMemRegion in) const noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t> > DecodeAndVerify(ReadOnlyMemRegion in) const noexcept = 0;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> DecodeAndVerify(ReadOnlyMem Region in) const noexcept;
    virtual netaos::core::Result<void> Reset() noexcept = 0;
    virtual netaos::core::Result<void> SetKey(const PublicKey& key) noexcept = 0;


   private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_MSG_RECOVERY_PUBLIC_CTX_H_