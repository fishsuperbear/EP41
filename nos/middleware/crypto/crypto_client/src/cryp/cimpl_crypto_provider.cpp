#include "cimpl_crypto_provider.h"

#include "client/crypto_cm_client.h"
#include "common/crypto_error_domain.h"
#include "common/crypto_logger.hpp"
#include "common/type_converter.h"
#include "cryp/cryobj/cimpl_symmetric_key.h"
#include "cryp/cryobj/cimpl_public_key.h"
#include "cryp/cryobj/cimpl_private_key.h"
#include "cryp/cimpl_decryptor_private_ctx.h"
#include "cryp/cimpl_encryptor_public_ctx.h"
#include "cryp/cimpl_signer_private_ctx.h"
#include "cryp/cimpl_symmetric_block_cipher_ctx.h"
#include "cryp/cimpl_verifier_public_ctx.h"
#include "keys/cimpl_io_interface.h"
// #include "imp_hash_function_ctx.h"
// #include "imp_volatile_trusted_containe.h"
// #include "imp_symmetric_block_cipher_ctx.h"
// #include "imp_decryptor_private_ctx.h"
// #include "imp_encryptor_public_ctx.h"
// #include "imp_private_key.h"
// #include "imp_signer_private_ctx.h"
// #include "imp_verifier_public_ctx.h"
// #include <memory>
// #include "openssl/rsa.h"
// #include "openssl/err.h"
// #include "openssl/evp.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {



CimplCryptoProvider::CimplCryptoProvider(){
    CRYP_INFO << "CimplCryptoProvider construct called.";
};

CimplCryptoProvider::~CimplCryptoProvider(){
    CRYP_INFO<<"CimplCryptoProvider destructed. called.";
    // CryptoCmClient::Instance().Stop();
};

netaos::core::Result<VolatileTrustedContainer::Uptr> CimplCryptoProvider::AllocVolatileContainer(std::size_t capacity) noexcept {

    // auto uptr = std::make_unique<ImpVolatileTrustedContainer>(capacity);
    // return netaos::core::Result<VolatileTrustedContainer::Uptr>::FromValue(std::move(uptr));
    return netaos::core::Result<VolatileTrustedContainer::Uptr>::FromError(CryptoErrc::kUnsupported);
};

netaos::core::Result<HashFunctionCtx::Uptr> CimplCryptoProvider::CreateHashFunctionCtx(AlgId algId) noexcept{
    // auto uptr = std::make_unique<ImpHashFunctionCtx>(algId);
    // return netaos::core::Result<HashFunctionCtx::Uptr>::FromValue(std::move(uptr));
    return netaos::core::Result<HashFunctionCtx::Uptr>::FromError(CryptoErrc::kUnsupported);
};

netaos::core::Result<SymmetricBlockCipherCtx::Uptr> CimplCryptoProvider::CreateSymmetricBlockCipherCtx(AlgId algId) noexcept{
    // auto uptr = std::make_unique<ImpSymmetricBlockCipherCtx>();
    // return netaos::core::Result<SymmetricBlockCipherCtx::Uptr>::FromValue(std::move(uptr));   

    CipherCtxRef ctx_ref;
    int32_t ipc_res = CryptoCmClient::Instance().CreateCipherContext(algId, kCipherContextType_SymmetricBlockCipher, ctx_ref);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        return netaos::core::Result<SymmetricBlockCipherCtx::Uptr>::FromError(static_cast<CryptoErrc>(ipc_res));
    }
    CRYP_INFO<<"CreateSymmetricBlockCipherCtx ctx_ref.ref:"<<ctx_ref.ref;

    auto uptr = std::make_unique<CimplSymmetricBlockCipherCtx>(ctx_ref);
    return netaos::core::Result<SymmetricBlockCipherCtx::Uptr>::FromValue(std::move(uptr));
};

netaos::core::Result<DecryptorPrivateCtx::Uptr> CimplCryptoProvider::CreateDecryptorPrivateCtx(AlgId algId) noexcept{
    CipherCtxRef ctx_ref;
    int32_t ipc_res = CryptoCmClient::Instance().CreateCipherContext(algId, kCipherContextType_DecryptorPrivate, ctx_ref);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        return netaos::core::Result<DecryptorPrivateCtx::Uptr>::FromError(static_cast<CryptoErrc>(ipc_res));
    }
    CRYP_INFO<<"CreateDecryptorPrivateCtx ctx_ref.ref:"<<ctx_ref.ref;

    auto uptr = std::make_unique<CimpDecryptorPrivateCtx>(ctx_ref);
    return netaos::core::Result<DecryptorPrivateCtx::Uptr>::FromValue(std::move(uptr));
};

netaos::core::Result<EncryptorPublicCtx::Uptr> CimplCryptoProvider::CreateEncryptorPublicCtx(AlgId algId) noexcept{
    // auto uptr = std::make_unique<ImpEncryptorPublicCtx>();
    // return netaos::core::Result<EncryptorPublicCtx::Uptr>::FromValue(std::move(uptr));   
    // return netaos::core::Result<EncryptorPublicCtx::Uptr>::FromError(CryptoErrc::kUnsupported);
    CipherCtxRef ctx_ref;
    int32_t ipc_res = CryptoCmClient::Instance().CreateCipherContext(algId, kCipherContextType_EncryptorPublic, ctx_ref);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        return netaos::core::Result<EncryptorPublicCtx::Uptr>::FromError(static_cast<CryptoErrc>(ipc_res));
    }
    CRYP_INFO<<"CreateEncryptorPublicCtx ctx_ref.ref:"<<ctx_ref.ref;

    auto uptr = std::make_unique<CimplEncryptorPublicCtx>(ctx_ref);
    return netaos::core::Result<EncryptorPublicCtx::Uptr>::FromValue(std::move(uptr));
};


netaos::core::Result<PrivateKey::Uptrc> CimplCryptoProvider::GeneratePrivateKey(AlgId algId, AllowedUsageFlags allowedUsage, bool isSession, bool isExportable) noexcept {
    // auto uptrc = std::make_unique<const ImpPrivateKey>(algId,allowedUsage,isSession,isExportable);
    // return netaos::core::Result<PrivateKey::Uptrc>::FromValue(std::move(uptrc));   
    // return netaos::core::Result<PrivateKey::Uptrc>::FromError(CryptoErrc::kUnsupported);
    CryptoKeyRef key_ref;
    int32_t ipc_res = CryptoCmClient::Instance().GenerateKey(kKeyType_PrivateKey, algId, allowedUsage, isSession, isExportable, key_ref);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        CRYP_ERROR << "Generate key failed. Crypto server replied " << CRYP_ERROR_MESSAGE(ipc_res);
        return netaos::core::Result<PrivateKey::Uptrc>::FromError(static_cast<CryptoErrc>(ipc_res));
    }

    // CryptoObject::CryptoObjectInfo object_info;
    // CryptoPrimitiveId::PrimitiveIdInfo primitive_id_info;
    // TypeConverter::CmStructToInnerType(key_ref.crypto_object_info(), object_info);
    // TypeConverter::CmStructToInnerType(key_ref.primitive_id_info(), primitive_id_info);
    CryptoPrimitiveId primitive_id(key_ref.primitive_id_info);
    CRYP_INFO<< "key_ref.ref:"<<key_ref.ref;
    auto uptrc = std::make_unique<const CimplPrivateKey>(key_ref.ref, key_ref.crypto_object_info, primitive_id, key_ref.allowed_usage);
    return netaos::core::Result<PrivateKey::Uptrc>::FromValue(std::move(uptrc));
};


netaos::core::Result<SymmetricKey::Uptrc> CimplCryptoProvider::GenerateSymmetricKey(AlgId algId, AllowedUsageFlags allowedUsage, bool isSession, bool isExportable)noexcept{
    CryptoKeyRef key_ref;
    int32_t ipc_res = CryptoCmClient::Instance().GenerateKey(kKeyType_SymmetricKey, algId, allowedUsage, isSession, isExportable, key_ref);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        CRYP_ERROR << "Generate key failed. Crypto server replied " << CRYP_ERROR_MESSAGE(ipc_res);
        return netaos::core::Result<SymmetricKey::Uptrc>::FromError(static_cast<CryptoErrc>(ipc_res));
    }

    // CryptoObject::CryptoObjectInfo object_info;
    // CryptoPrimitiveId::PrimitiveIdInfo primitive_id_info;
    // TypeConverter::CmStructToInnerType(key_ref.crypto_object_info(), object_info);
    // TypeConverter::CmStructToInnerType(key_ref.primitive_id_info(), primitive_id_info);
    CryptoPrimitiveId primitive_id(key_ref.primitive_id_info);
    CRYP_INFO<< "key_ref.ref:"<<key_ref.ref;
    auto uptrc = std::make_unique<const CimplSymmetricKey>(key_ref.ref, key_ref.crypto_object_info, primitive_id, key_ref.allowed_usage);
    return netaos::core::Result<SymmetricKey::Uptrc>::FromValue(std::move(uptrc));
};

// CimplCryptoProvider&  CimplCryptoProvider::Instance(){
//     if(!instance_){
//         std::lock_guard<std::mutex> lock(instance_mutex_);
//         instance_ = new CimplCryptoProvider;
//     }
//     return *instance_;
// }

netaos::core::Result<SignerPrivateCtx::Uptr> CimplCryptoProvider::CreateSignerPrivateCtx(AlgId algId) noexcept{
    CipherCtxRef ctx_ref;
    int32_t ipc_res = CryptoCmClient::Instance().CreateCipherContext(algId, kCipherContextType_SignerPrivate, ctx_ref);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        return netaos::core::Result<SignerPrivateCtx::Uptr>::FromError(static_cast<CryptoErrc>(ipc_res));
    }
    CRYP_INFO<<"CreateSignerPrivateCtx ctx_ref.ref:"<<ctx_ref.ref;

    auto uptr = std::make_unique<CimplSignerPrivateCtx>(ctx_ref);
    return netaos::core::Result<SignerPrivateCtx::Uptr>::FromValue(std::move(uptr));
};

netaos::core::Result<VerifierPublicCtx::Uptr> CimplCryptoProvider::CreateVerifierPublicCtx(AlgId algId) noexcept{
    CipherCtxRef ctx_ref;
    int32_t ipc_res = CryptoCmClient::Instance().CreateCipherContext(algId, kCipherContextType_VerifierPublic, ctx_ref);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        return netaos::core::Result<VerifierPublicCtx::Uptr>::FromError(static_cast<CryptoErrc>(ipc_res));
    }
    CRYP_INFO<<"CreateVerifierPublicCtx ctx_ref.ref:"<<ctx_ref.ref;

    auto uptr = std::make_unique<CimplVerifierPublicCtx>(ctx_ref);
    return netaos::core::Result<VerifierPublicCtx::Uptr>::FromValue(std::move(uptr));
}


CryptoProvider::AlgId CimplCryptoProvider::ConvertToAlgId(netaos::core::StringView primitiveName) const noexcept {
    CryptoProvider::AlgId ret = kAlgIdUndefined ;
    netaos::core::String name(primitiveName.data());
    for(auto it =toAlgNameMap_.begin();it != toAlgNameMap_.end();it++){
        if(!name.compare(it->second)){
            ret = it->first;
            break;
        }
    }
   return ret;
};

netaos::core::Result<netaos::core::String> CimplCryptoProvider::ConvertToAlgName(AlgId algId) const noexcept {
    netaos::core::String ret;
    auto it = toAlgNameMap_.find(algId);
    if(it != toAlgNameMap_.end()){
        ret = it->second;
    }else{

    }
   return netaos::core::Result<netaos::core::String>::FromValue(std::move(ret));
};

netaos::core::Result<PrivateKey::Uptrc> CimplCryptoProvider::LoadPrivateKey(const IOInterface& container) noexcept {
    // std::vector<std::uint8_t> payload =  container.GetPayload();
    // EVP_PKEY *pkey = EVP_PKEY_new();
    // const std::uint8_t *pdata = payload.data();
    // if(!payload.empty()){
    //     d2i_PrivateKey(EVP_PKEY_RSA , &pkey, &pdata,payload.size());
    // }

    // ImpPrivateKey::Uptrc uptrc = std::make_unique<const ImpPrivateKey>(container.GetPrimitiveId(),container.GetAllowedUsage(),container.IsObjectSession(),container.IsObjectExportable(),pkey);
    // return netaos::core::Result<PrivateKey::Uptrc>::FromValue(std::move(uptrc));
    CryptoKeyRef key_ref;
    int32_t ipc_res = CryptoCmClient::Instance().LoadPrivateKey(dynamic_cast<CimplOInterface*>(const_cast<IOInterface*>(&container))->getContainer().ref, key_ref);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        CRYP_ERROR << "LoadPrivateKey failed. Crypto server replied " << CRYP_ERROR_MESSAGE(ipc_res);
        return netaos::core::Result<PrivateKey::Uptrc>::FromError(static_cast<CryptoErrc>(ipc_res));
    }
    CryptoPrimitiveId primitive_id(key_ref.primitive_id_info);
    CRYP_INFO<< "key_ref.ref:"<<key_ref.ref;
    auto uptrc = std::make_unique<const CimplPrivateKey>(key_ref.ref, key_ref.crypto_object_info, primitive_id, key_ref.allowed_usage);
    return netaos::core::Result<PrivateKey::Uptrc>::FromValue(std::move(uptrc));
};


bool CimplCryptoProvider::Init()  {
    return CryptoCmClient::Instance().Init();
}

bool CimplCryptoProvider::Deinit()  {
    // return CryptoCmClient::Instance().Deinit();
    return true;
}


// netaos::core::Result<PublicKey::Uptrc> LoadPublicKey(const IOInterface& container) noexcept {
    
// };

}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara
