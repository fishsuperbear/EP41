#include "cimpl_verifier_public_ctx.h"

#include "client/crypto_cm_client.h"
#include "common/crypto_logger.hpp"
#include "cryp/cimpl_crypto_provider.h"
#include "cryp/cryobj/cimpl_public_key.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

CimplVerifierPublicCtx::CimplVerifierPublicCtx(AlgId id)
: alg_id_(id) {

}

CimplVerifierPublicCtx::CimplVerifierPublicCtx(const CipherCtxRef& ctx_ref)
: ctx_ref_(ctx_ref) {

}

CimplVerifierPublicCtx::~CimplVerifierPublicCtx() {
  CryptoCmClient::Instance().ReleaseObject(ctx_ref_.ref);
}

netaos::core::Result<void> CimplVerifierPublicCtx::Reset() noexcept {
  return netaos::core::Result<void>();
}

netaos::core::Result<void> CimplVerifierPublicCtx::SetKey(const PublicKey& key) noexcept {
  if (ctx_ref_.ref == 0x0u) {
    CRYP_INFO<< "ctx_ref_.ref is null.";
    return netaos::core::Result<void>::FromError(CryptoErrc::kInvalidUsageOrder);
  }
  ctx_ref_.ctx_type = kCipherContextType_VerifierPublic;

  const CimplPublicKey& cimpl_sym_key = dynamic_cast<const CimplPublicKey&>(key);
  const CryptoKeyRef key_ref = cimpl_sym_key.GetKeyRef();
  CRYP_INFO<<"SetKey:begin ContextSetKey.";
  int32_t transform = 0;
  int32_t ipc_res = CryptoCmClient::Instance().ContextSetKey(ctx_ref_, key_ref, static_cast<int32_t>(transform));
  if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
    return netaos::core::Result<void>::FromError(static_cast<CryptoErrc>(ipc_res));
  }

  ctx_ref_.transform = static_cast<int32_t>(transform);
  isInitialized_ = true;
  return netaos::core::Result<void>();
}

netaos::core::Result<bool> CimplVerifierPublicCtx::Verify(ReadOnlyMemRegion value,
      ReadOnlyMemRegion signature, ReadOnlyMemRegion context) const noexcept {

  bool ret = false;

  if (ctx_ref_.ref == 0x0u) {
    CRYP_INFO<< "ProcessBlock ctx_ref_.ref is null.";
    return netaos::core::Result<bool>::FromValue(ret);
  }
  ctx_ref_.ctx_type = kCipherContextType_VerifierPublic;
  std::vector<uint8_t> in_vec;
  in_vec.resize(value.size());
  memcpy(in_vec.data(), value.data(), value.size());
  std::vector<uint8_t> out_vec;
  out_vec.resize(signature.size());
  memcpy(out_vec.data(), signature.data(), signature.size());
  int32_t ipc_res = CryptoCmClient::Instance().CryptoTrans(ctx_ref_, in_vec, out_vec, false);
  if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
    ret = false;
  } else {
    ret = true;
  }

  return netaos::core::Result<bool>::FromValue(ret);

}

bool CimplVerifierPublicCtx::IsInitialized() const noexcept{
    return isInitialized_;
}

CryptoPrimitiveId::Uptr CimplVerifierPublicCtx::GetCryptoPrimitiveId() const noexcept{
    auto uptr = std::make_unique<CryptoPrimitiveId>();
    return uptr;
}

CryptoProvider& CimplVerifierPublicCtx::MyProvider() const noexcept{
    CryptoProvider* prov = new CimplCryptoProvider;
    return *prov;
}

}  // namespace cryp
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon