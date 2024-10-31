#include "cimpl_encryptor_public_ctx.h"

#include <memory>

#include "openssl/err.h"
#include "openssl/evp.h"
// #include "openssl/types.h"
#include "openssl/rsa.h"

#include "cryp/cimpl_crypto_provider.h"
#include "cryp/imp_crypto_service.h"
#include "cryp/cryobj/cimpl_public_key.h"
#include "common/crypto_error_domain.h"
#include "common/crypto_logger.hpp"
#include "client/crypto_cm_client.h"


namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

static const unsigned char aes_key[] = "0123456789abcdeF";

CimplEncryptorPublicCtx::CimplEncryptorPublicCtx(const CipherCtxRef& ctx_ref)
: ctx_ref_(ctx_ref) {

}

CimplEncryptorPublicCtx::~CimplEncryptorPublicCtx() {
    CryptoCmClient::Instance().ReleaseObject(ctx_ref_.ref);
}

CryptoService::Uptr CimplEncryptorPublicCtx::GetCryptoService() const noexcept{
    return std::make_unique<ImpCryptoService>();
}

netaos::core::Result<netaos::core::Vector<uint8_t>> CimplEncryptorPublicCtx::ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding) const noexcept {
    if (ctx_ref_.ref == 0x0u) {
        CRYP_INFO<< "ProcessBlock ctx_ref_.ref is null.";
        return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromError(CryptoErrc::kInvalidUsageOrder);
    }
    ctx_ref_.ctx_type = kCipherContextType_EncryptorPublic;
    std::vector<uint8_t> in_vec;
    in_vec.resize(in.size());
    memcpy(in_vec.data(), in.data(), in.size());
    std::vector<uint8_t> out_vec;
    int32_t ipc_res = CryptoCmClient::Instance().CryptoTrans(ctx_ref_, in_vec, out_vec, suppressPadding);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromError(static_cast<CryptoErrc>(ipc_res));
    }

    return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromValue(std::move(out_vec));
}

netaos::core::Result<void> CimplEncryptorPublicCtx::Reset() noexcept{
    return netaos::core::Result<void>();
}

netaos::core::Result<void> CimplEncryptorPublicCtx::SetKey(const PublicKey& key) noexcept{
   if (ctx_ref_.ref == 0x0u) {
        CRYP_INFO<< "ctx_ref_.ref is null.";
        return netaos::core::Result<void>::FromError(CryptoErrc::kInvalidUsageOrder);
    }
    ctx_ref_.ctx_type = kCipherContextType_EncryptorPublic;

    const CimplPublicKey& cimpl_pub_key = dynamic_cast<const CimplPublicKey&>(key);
    const CryptoKeyRef key_ref = cimpl_pub_key.GetKeyRef();
    CRYP_INFO<<"SetKey:begin ContextSetKey.";
    int32_t ipc_res = CryptoCmClient::Instance().ContextSetKey(ctx_ref_, key_ref, 0);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        CRYP_INFO<< "CimplEncryptorPublicCtx SetKey error: ";
        return netaos::core::Result<void>::FromError(static_cast<CryptoErrc>(ipc_res));
    }

    // ctx_ref_.transform = static_cast<int32_t>(transform);
    isInitialized_ = true;
    return netaos::core::Result<void>();
}

CryptoPrimitiveId::Uptr CimplEncryptorPublicCtx::GetCryptoPrimitiveId () const noexcept {
    return CryptoPrimitiveId::Uptr(new CryptoPrimitiveId(alg_id_));
}

bool CimplEncryptorPublicCtx::IsInitialized() const noexcept{
    return isInitialized_;
}

CryptoProvider& CimplEncryptorPublicCtx::MyProvider() const noexcept{
    CryptoProvider* prov = new CimplCryptoProvider;
    return *prov;
}

// ImpSymmetricBlockCipherCtx::ImpSymmetricBlockCipherCtx(){

// }

}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara
