#include "cimpl_signer_private_ctx.h"

#include <iostream>
#include <cstddef>
#include "openssl/rsa.h"

// #include "common/imp_volatile_trusted_container.h"
#include "common/crypto_logger.hpp"
#include "cryp/cimpl_crypto_provider.h"
#include "cryp/cryobj/cimpl_private_key.h"
// #include "cryp/imp_hash_function_ctx.h"
#include "client/crypto_cm_client.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

CimplSignerPrivateCtx::CimplSignerPrivateCtx(const CipherCtxRef& ctx_ref)
: ctx_ref_(ctx_ref) {

}

CimplSignerPrivateCtx::~CimplSignerPrivateCtx() {
    CryptoCmClient::Instance().ReleaseObject(ctx_ref_.ref);
}

netaos::core::Result<void> CimplSignerPrivateCtx::Reset() noexcept {
    return netaos::core::Result<void>();
}

netaos::core::Result<void> CimplSignerPrivateCtx::SetKey(const PrivateKey& key) noexcept {
    if (ctx_ref_.ref == 0x0u) {
        CRYP_INFO<< "ctx_ref_.ref is null.";
        return netaos::core::Result<void>::FromError(CryptoErrc::kInvalidUsageOrder);
    }
    ctx_ref_.ctx_type = kCipherContextType_SignerPrivate;

    const CimplPrivateKey& cimpl_sym_key = dynamic_cast<const CimplPrivateKey&>(key);
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

netaos::core::Result<netaos::core::Vector<uint8_t>> CimplSignerPrivateCtx::Sign(ReadOnlyMemRegion value, ReadOnlyMemRegion context) const noexcept{
    if (ctx_ref_.ref == 0x0u) {
        CRYP_INFO<< "ProcessBlock ctx_ref_.ref is null.";
        return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromError(CryptoErrc::kInvalidUsageOrder);
    }
    ctx_ref_.ctx_type = kCipherContextType_SignerPrivate;
    std::vector<uint8_t> in_vec;
    in_vec.resize(value.size());
    memcpy(in_vec.data(), value.data(), value.size());
    std::vector<uint8_t> out_vec;
    int32_t ipc_res = CryptoCmClient::Instance().CryptoTrans(ctx_ref_, in_vec, out_vec, false);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromError(static_cast<CryptoErrc>(ipc_res));
    }

    return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromValue(std::move(out_vec));
}


netaos::core::Result<netaos::core::Vector<uint8_t>> CimplSignerPrivateCtx::SignPreHashed (const HashFunctionCtx &hashFn, ReadOnlyMemRegion context) const noexcept{
    std::vector<uint8_t> ret;
    #if 0
    std::string signature;
    std::string hash;
    unsigned int sigLen = 0;
    RSA *rsa = NULL;
    int mdType = 0;

    auto digestResult = hashFn.GetDigest();
    if(digestResult.HasValue()){
        hash.resize(digestResult.Value().size());
        for(std::size_t i=0;i<digestResult.Value().size();i++){
            hash[i]  = static_cast<unsigned char>(digestResult.Value()[i]);
        }
    }else{
        CRYP_ERROR<<"digestResult is NULL.";
        // return ;  //return error
    }

    ImpPrivateKey *imp_private_key =dynamic_cast<ImpPrivateKey*>(pprivate_key_);
    ImpHashFunctionCtx *impHashFn =const_cast<ImpHashFunctionCtx*>(dynamic_cast<const ImpHashFunctionCtx*>(&hashFn));

    rsa = EVP_PKEY_get1_RSA(imp_private_key->get_pkey());

    switch (impHashFn->GetCryptoPrimitiveId()->GetPrimitiveId())
    {
    case kAlgIdSHA256:
        mdType = NID_sha256;
        break;
    case kAlgIdSHA384:
        mdType = NID_sha384;
        break;
    case kAlgIdSHA512:
        mdType = NID_sha384;
        break;
    default:
        break;
    }

    signature.resize(RSA_size(rsa));
    int result = RSA_sign(mdType,reinterpret_cast<const unsigned char*>(hash.data()),hash.size(),reinterpret_cast<unsigned char*>(signature.data()),&sigLen,rsa);
    if (!result) {
        CRYP_ERROR<<"RSA_sign failed.";
        // return ;  //return error
    }else{
        CRYP_INFO<<"RSA_sign size:"<<sigLen;

    }
    signature.resize(sigLen);
    ret.resize(sigLen);
    for (std::size_t i = 0; i < signature.size(); i++) {
        ret[i] = static_cast<uint8_t>(signature[i]);
    }
    #endif
    return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromValue(std::move(ret));
}


netaos::core::Result<netaos::core::Vector<uint8_t>> CimplSignerPrivateCtx::SignPreHashed(AlgId hashAlgId, ReadOnlyMemRegion hashValue, ReadOnlyMemRegion context) const noexcept {

    if (ctx_ref_.ref == 0x0u) {
        CRYP_INFO<< "ProcessBlock ctx_ref_.ref is null.";
        return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromError(CryptoErrc::kInvalidUsageOrder);
    }
    ctx_ref_.ctx_type = kCipherContextType_SignPrehash;
    ctx_ref_.alg_id = hashAlgId;
    std::vector<uint8_t> in_vec;
    in_vec.resize(hashValue.size());
    memcpy(in_vec.data(), hashValue.data(), hashValue.size());
    std::vector<uint8_t> out_vec;
    int32_t ipc_res = CryptoCmClient::Instance().CryptoTrans(ctx_ref_, in_vec, out_vec, false);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromError(static_cast<CryptoErrc>(ipc_res));
    }

    return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromValue(std::move(out_vec));
}

CryptoPrimitiveId::Uptr CimplSignerPrivateCtx::GetCryptoPrimitiveId () const noexcept {
    return CryptoPrimitiveId::Uptr(new CryptoPrimitiveId(alg_id_));
}

bool CimplSignerPrivateCtx::IsInitialized() const noexcept{
    return isInitialized_;
}

// CryptoPrimitiveId::Uptr CimplSignerPrivateCtx::GetCryptoPrimitiveId() const noexcept{
//     auto uptr = std::make_unique<CryptoPrimitiveId>();
//     return uptr;
// }



CryptoProvider& CimplSignerPrivateCtx::MyProvider() const noexcept{
    // return const_cast<ImpCryptoProvider&>(ImpCryptoProvider::Instance());
    // CryptoProvider& prov = ImpCryptoProvider::Instance();
    CryptoProvider* prov = new CimplCryptoProvider;
    return *prov;
}


netaos::core::Result<PublicKey::Uptrc> CimplSignerPrivateCtx::GetPublicKey() const noexcept{
    PublicKey::Uptrc uptrc = NULL;
    if(pprivate_key_){
        if(pprivate_key_->GetPublicKey().HasValue()){
            return netaos::core::Result<PublicKey::Uptrc>::FromValue(std::move(pprivate_key_->GetPublicKey().Value()));
        }else{
            // return netaos::core::Result<PublicKey::Uptrc>::FromError(CryptoErrc::kLogicFault);
            return netaos::core::Result<PublicKey::Uptrc>::FromValue(std::move(uptrc));

        }
    }else{
        // return netaos::core::Result<PublicKey::Uptrc>::FromError(CryptoErrc::kLogicFault);
        return netaos::core::Result<PublicKey::Uptrc>::FromValue(std::move(uptrc));
    }
}

}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara
