#include "cryp/imp_signer_private_ctx.h"

#include <iostream>
#include <cstddef>
#include "openssl/rsa.h"

#include "common/imp_volatile_trusted_container.h"
#include "common/crypto_logger.hpp"
#include "common/memory_utility.h"
#include "cryp/imp_crypto_provider.h"
#include "cryp/cryobj/imp_private_key.h"
#include "cryp/imp_hash_function_ctx.h"
#include "cryp/crypto_provider.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

netaos::core::Result<void> ImpSignerPrivateCtx::Reset() noexcept {
    return netaos::core::Result<void>();
}

netaos::core::Result<void> ImpSignerPrivateCtx::SetKey(const PrivateKey& key) noexcept {
    //上下文初始化
    std::string hash_name;
    openssl_ctx_.plib_ctx = OSSL_LIB_CTX_new();
    if (!openssl_ctx_.plib_ctx) {
        CRYP_ERROR<<"OSSL_LIB_CTX_new() returned NULL.";
    }
    switch (alg_id_)
    {
    case kAlgIdRSA2048SHA384PSS:
        hash_name = "SHA2-384";
        break;
    case kAlgIdRSA2048SHA512PSS:
        hash_name = "SHA2-512";
        break;
    case kAlgIdRSA2048SHA256PSS:
        hash_name = "SHA2-256";
        break;
    default:
        break;
    }
    hash_name = "SHA2-256";
    CRYP_INFO << "ImpSignerPrivateCtx SetKey hash_name : " << hash_name;
    openssl_ctx_.pmd = EVP_MD_fetch(openssl_ctx_.plib_ctx,static_cast<const char *>(hash_name.data()), NULL);
    if (!openssl_ctx_.pmd) {
        CRYP_ERROR<< "EVP_MD_fetch could not find hash name.";
    }

    openssl_ctx_.pmd_ctx = EVP_MD_CTX_new();
    if (!openssl_ctx_.pmd_ctx) {
        CRYP_ERROR<<"EVP_MD_CTX_new failed.";
    }

    if(EVP_SignInit(openssl_ctx_.pmd_ctx, openssl_ctx_.pmd)) {
        pprivate_key_ = const_cast<PrivateKey *>(&key);
        isInitialized_ = true;
    }else{
        isInitialized_ = false;
        CRYP_ERROR<<"EVP_SignInit failed.";
    }

    return netaos::core::Result<void>();

}

netaos::core::Result<netaos::core::Vector<uint8_t>> ImpSignerPrivateCtx::Sign(ReadOnlyMemRegion value, ReadOnlyMemRegion context) const noexcept{
    std::vector<uint8_t> out;
    //Determine the length of the fetched digest type 
    // openssl_ctx_.sign_len = EVP_MD_get_size(openssl_ctx_.pmd);
    // if (openssl_ctx_.sign_len <= 0) {
    //     CRYP_ERROR<<"EVP_MD_get_size returned invalid size.";
    // }else{
    //     CRYP_ERROR<<"EVP_MD_get_size: "<<openssl_ctx_.sign_len;
    // }

    if( EVP_SignUpdate(openssl_ctx_.pmd_ctx, value.data(), value.size()) ) {
        CRYP_INFO<< "EVP_SignUpdate success.";
    }else{
        CRYP_ERROR<< "EVP_SignUpdate failed.";
    }

    ImpPrivateKey *imp_private_key =dynamic_cast<ImpPrivateKey*>(pprivate_key_); //ok

    // Get buffer length for the signature.
    unsigned int sig_len = 0;
    if (0 == EVP_SignFinal(openssl_ctx_.pmd_ctx, nullptr, &sig_len, imp_private_key->get_pkey())) {
        CRYP_ERROR << "Get signature length failed by EVP_SignFinal";
    }
    CRYP_INFO << "Expected signature length: " << sig_len;

    // Make openssl managed memory for signature.
    hozon::netaos::crypto::OsslMemSptr sig_buf = MakeSharedOsslMem(sig_len);
    if (!sig_buf) {
        CRYP_ERROR<< "Memory cannnot be allocated for signautre";
    }

    // Finalize sign operation and get the signature.
    if(EVP_SignFinal(openssl_ctx_.pmd_ctx, sig_buf.get(), &sig_len, imp_private_key->get_pkey())){
            out.resize(sig_len);
            memcpy(out.data(), sig_buf.get(), sig_len);
    }else{
        CRYP_ERROR<<"EVP_SignFinal() failed.";

    }
    return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromValue(std::move(out));
}


netaos::core::Result<netaos::core::Vector<uint8_t>> ImpSignerPrivateCtx::SignPreHashed (const HashFunctionCtx &hashFn, ReadOnlyMemRegion context) const noexcept{
    std::vector<uint8_t> ret;
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
    return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromValue(std::move(ret));
}


netaos::core::Result<netaos::core::Vector<uint8_t>> ImpSignerPrivateCtx::SignPreHashed(AlgId hashAlgId, ReadOnlyMemRegion hashValue, ReadOnlyMemRegion context) const noexcept{
    std::vector<uint8_t> ret;
    std::string signature;
    std::string hash;
    unsigned int sigLen = 0;
    RSA *rsa = NULL;
    int mdType = 0;



    if (hashValue.empty()){
        CRYP_ERROR<<"hashValue is NULL.";
        //return error
    }

    switch (hashAlgId)
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

    ImpPrivateKey *imp_private_key =dynamic_cast<ImpPrivateKey*>(pprivate_key_); 
    rsa = EVP_PKEY_get1_RSA(imp_private_key->get_pkey());

    signature.resize(RSA_size(rsa));
    int result = RSA_sign(mdType,static_cast<const unsigned char*>(hashValue.data()),hashValue.size(), reinterpret_cast<unsigned char*>(signature.data()),&sigLen,rsa);
    if (!result) {
        CRYP_ERROR<<"RSA_sign failed.";
    }else{
        CRYP_INFO<<"RSA_sign size:"<<sigLen;
    }
    signature.resize(sigLen);
    ret.resize(sigLen);
    for (std::size_t i = 0; i < signature.size(); i++) {
        ret[i] = static_cast<uint8_t>(signature[i]);
    }
    return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromValue(std::move(ret));
}

CryptoPrimitiveId::Uptr ImpSignerPrivateCtx::GetCryptoPrimitiveId () const noexcept {
    return CryptoPrimitiveId::Uptr(new CryptoPrimitiveId(alg_id_));
}

bool ImpSignerPrivateCtx::IsInitialized() const noexcept{
    return isInitialized_;
}

// CryptoPrimitiveId::Uptr ImpSignerPrivateCtx::GetCryptoPrimitiveId() const noexcept{
//     auto uptr = std::make_unique<CryptoPrimitiveId>();
//     return uptr;
// }



CryptoProvider& ImpSignerPrivateCtx::MyProvider() const noexcept{
    // return const_cast<ImpCryptoProvider&>(ImpCryptoProvider::Instance());
    // CryptoProvider& prov = ImpCryptoProvider::Instance();
    CryptoProvider* prov = new ImpCryptoProvider;
    return *prov;
}


netaos::core::Result<PublicKey::Uptrc> ImpSignerPrivateCtx::GetPublicKey() const noexcept{
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
