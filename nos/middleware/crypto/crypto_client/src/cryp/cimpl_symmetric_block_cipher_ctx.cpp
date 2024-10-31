#include "cimpl_symmetric_block_cipher_ctx.h"

#include <memory>

#include "common/crypto_error_domain.h"
#include "common/crypto_logger.hpp"
#include "cryp/cimpl_crypto_provider.h"
#include "cryp/imp_crypto_service.h"
#include "cryp/cryobj/cimpl_symmetric_key.h"
#include "client/crypto_cm_client.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

static const unsigned char aes_key[] = "0123456789abcdeF";

CimplSymmetricBlockCipherCtx::CimplSymmetricBlockCipherCtx(const CipherCtxRef& ctx_ref)
: ctx_ref_(ctx_ref) {

}

CimplSymmetricBlockCipherCtx::~CimplSymmetricBlockCipherCtx() {
    CryptoCmClient::Instance().ReleaseObject(ctx_ref_.ref);
}

CryptoService::Uptr CimplSymmetricBlockCipherCtx::GetCryptoService() const noexcept{
    return std::make_unique<ImpCryptoService>();
}

netaos::core::Result<CryptoTransform> CimplSymmetricBlockCipherCtx::GetTransformation() const noexcept {
    return netaos::core::Result<CryptoTransform>::FromValue(static_cast<CryptoTransform>(ctx_ref_.transform));
}

netaos::core::Result<netaos::core::Vector<uint8_t>> CimplSymmetricBlockCipherCtx::ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding) const noexcept {

    if (ctx_ref_.ref == 0x0u) {
        CRYP_INFO<< "ProcessBlock ctx_ref_.ref is null.";
        return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromError(CryptoErrc::kInvalidUsageOrder);
    }
    ctx_ref_.ctx_type = kCipherContextType_SymmetricBlockCipher;
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

netaos::core::Result<netaos::core::Vector<uint8_t>> CimplSymmetricBlockCipherCtx::ProcessBlocks(ReadOnlyMemRegion in) const noexcept{
    return ProcessBlock(in, true);
}

netaos::core::Result<void> CimplSymmetricBlockCipherCtx::Reset() noexcept{
    return netaos::core::Result<void>::FromError(CryptoErrc::kUnsupported);
}

netaos::core::Result<void> CimplSymmetricBlockCipherCtx::SetKey(const SymmetricKey& key, CryptoTransform transform) noexcept{

    if (ctx_ref_.ref == 0x0u) {
        CRYP_INFO<< "ctx_ref_.ref is null.";
        return netaos::core::Result<void>::FromError(CryptoErrc::kInvalidUsageOrder);
    }
    ctx_ref_.ctx_type = kCipherContextType_SymmetricBlockCipher;

    const CimplSymmetricKey& cimpl_sym_key = dynamic_cast<const CimplSymmetricKey&>(key);
    const CryptoKeyRef key_ref = cimpl_sym_key.GetKeyRef();
    CRYP_INFO<<"SetKey:begin ContextSetKey.";
    int32_t ipc_res = CryptoCmClient::Instance().ContextSetKey(ctx_ref_, key_ref, static_cast<int32_t>(transform));
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        return netaos::core::Result<void>::FromError(static_cast<CryptoErrc>(ipc_res));
    }

    ctx_ref_.transform = static_cast<int32_t>(transform);
    isInitialized_ = true;
    return netaos::core::Result<void>();
}

CryptoPrimitiveId::Uptr CimplSymmetricBlockCipherCtx::GetCryptoPrimitiveId () const noexcept {
    return CryptoPrimitiveId::Uptr(new CryptoPrimitiveId(ctx_ref_.alg_id));
}

bool CimplSymmetricBlockCipherCtx::IsInitialized() const noexcept{
    return isInitialized_;
}

CryptoProvider& CimplSymmetricBlockCipherCtx::MyProvider() const noexcept{
    CryptoProvider* prov = new CimplCryptoProvider;
    return *prov;
}

}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara
