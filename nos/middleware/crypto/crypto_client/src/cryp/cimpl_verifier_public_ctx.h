#ifndef ARA_CRYPTO_CRYP_IMP_VERIFIER_PUBLIC_CTX_H_
#define ARA_CRYPTO_CRYP_IMP_VERIFIER_PUBLIC_CTX_H_

#include "common/inner_types.h"
#include "cryp/verifier_public_ctx.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class CimplVerifierPublicCtx : public VerifierPublicCtx
{

public:
  using Uptr = std::unique_ptr<CimplVerifierPublicCtx>;
  CimplVerifierPublicCtx(AlgId id);
  CimplVerifierPublicCtx(const CipherCtxRef& ctx_ref);
  ~CimplVerifierPublicCtx();
  netaos::core::Result<void> Reset() noexcept override;
  netaos::core::Result<void> SetKey(const PublicKey& key) noexcept override;
  netaos::core::Result<bool> Verify(ReadOnlyMemRegion value,
                                    ReadOnlyMemRegion signature,
                                    ReadOnlyMemRegion context =
                                    ReadOnlyMemRegion()) const noexcept override;
  CryptoPrimitiveId::Uptr GetCryptoPrimitiveId () const noexcept override;
  bool IsInitialized () const noexcept override;
  CryptoProvider& MyProvider() const noexcept override;

private:
  CryptoAlgId alg_id_ = kAlgIdUndefined;
  bool isInitialized_ = false;
  mutable CipherCtxRef ctx_ref_;
};

}  // namespace cryp
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon

#endif  // #ARA_CRYPTO_CRYP_IMP_VERIFIER_PUBLIC_CTX_H_