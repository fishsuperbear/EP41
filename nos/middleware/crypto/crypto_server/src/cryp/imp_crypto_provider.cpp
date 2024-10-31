#include "cryp/imp_crypto_provider.h"

#include <memory>
#include <openssl/rsa.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#include "common/crypto_logger.hpp"
#include "common/imp_volatile_trusted_container.h"
#include "common/uuid.h"
#include "cryp/imp_hash_function_ctx.h"
#include "cryp/imp_symmetric_block_cipher_ctx.h"
#include "cryp/imp_decryptor_private_ctx.h"
#include "cryp/imp_encryptor_public_ctx.h"
#include "cryp/cryobj/imp_private_key.h"
#include "cryp/imp_signer_private_ctx.h"
#include "cryp/imp_verifier_public_ctx.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {



ImpCryptoProvider::ImpCryptoProvider(){

}

netaos::core::Result<VolatileTrustedContainer::Uptr> ImpCryptoProvider::AllocVolatileContainer(std::size_t capacity) noexcept {

    auto uptr = std::make_unique<ImpVolatileTrustedContainer>(capacity);
    return netaos::core::Result<VolatileTrustedContainer::Uptr>::FromValue(std::move(uptr));
}

netaos::core::Result<HashFunctionCtx::Uptr> ImpCryptoProvider::CreateHashFunctionCtx(AlgId algId) noexcept{
    // if(algId == kAlgIdMD5 || algId == kAlgIdSHA1 || algId == kAlgIdSHA256 || algId == kAlgIdSHA384 || algId == kAlgIdSHA512){
    //     auto uptr = std::make_unique<ImpHashFunctionCtx>();
    //     return uptr;
    // }
    auto uptr = std::make_unique<ImpHashFunctionCtx>(algId);
    return netaos::core::Result<HashFunctionCtx::Uptr>::FromValue(std::move(uptr));
}

netaos::core::Result<SymmetricBlockCipherCtx::Uptr> ImpCryptoProvider::CreateSymmetricBlockCipherCtx(AlgId algId) noexcept{
    auto uptr = std::make_unique<ImpSymmetricBlockCipherCtx>(algId);
    return netaos::core::Result<SymmetricBlockCipherCtx::Uptr>::FromValue(std::move(uptr));   
}

netaos::core::Result<DecryptorPrivateCtx::Uptr> ImpCryptoProvider::CreateDecryptorPrivateCtx(AlgId algId) noexcept{
    auto uptr = std::make_unique<ImpDecryptorPrivateCtx>();
    return netaos::core::Result<DecryptorPrivateCtx::Uptr>::FromValue(std::move(uptr));   
}

netaos::core::Result<EncryptorPublicCtx::Uptr> ImpCryptoProvider::CreateEncryptorPublicCtx(AlgId algId) noexcept{
    auto uptr = std::make_unique<ImpEncryptorPublicCtx>();
    return netaos::core::Result<EncryptorPublicCtx::Uptr>::FromValue(std::move(uptr));   
}


netaos::core::Result<PrivateKey::Uptrc> ImpCryptoProvider::GeneratePrivateKey(AlgId algId, AllowedUsageFlags allowedUsage, bool isSession, bool isExportable) noexcept {
    std::string hash_name = "";
    switch (algId)
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
    CRYP_INFO << "ImpCryptoProvider GeneratePrivateKey algId hash_name: "<< hash_name;
    CryptoPrimitiveId primitive_id(algId);
    CryptoObject::CryptoObjectInfo crypto_object_info;
    crypto_object_info.isSession = isSession;
    crypto_object_info.isExportable = isExportable;
    crypto_object_info.objectUid.mCOType = CryptoObjectType::kPrivateKey;
    auto uuid_res = MakeVersion4Uuid();
    if (!uuid_res) {
        CRYP_ERROR << "Make version 4 uuid faild. " << CRYP_ERROR_MESSAGE(uuid_res.Error().Value());
        return netaos::core::Result<PrivateKey::Uptrc>::FromError(uuid_res.Error());
    }
    CRYP_INFO << "MakeVersion4Uuid finish. ";
    crypto_object_info.objectUid.mCouid = { uuid_res.Value(), 4 };
    crypto_object_info.dependencyUid.mCOType = CryptoObjectType::kPrivateKey;
    crypto_object_info.dependencyUid.mCouid = CryptoObjectUid { Uuid {0, 0}, 4 };
    crypto_object_info.isSession = isSession;
    crypto_object_info.isExportable = isExportable;
    CRYP_INFO << " make ImpPrivateKey uptr begin. ";
    auto uptrc = std::make_unique<const ImpPrivateKey>(crypto_object_info, primitive_id, allowedUsage);
    CRYP_INFO << " make ImpPrivateKey uptr finish. ";
    return netaos::core::Result<PrivateKey::Uptrc>::FromValue(std::move(uptrc));   
}

int ImpCryptoProvider::getCipherKeyLength(AlgId algId) {
    int keyLen;
    switch (algId)
    {
    case kAlgIdCBCAES128:
    case kAlgIdECBAES128:
    case kAlgIdGCMAES128:
    {
        keyLen = EVP_CIPHER_key_length(EVP_aes_128_ecb());
    }
        break;
    case kAlgIdCBCAES192:
    case kAlgIdGCMAES192:
    {
        keyLen = EVP_CIPHER_key_length(EVP_aes_192_ecb());
    }
        break;
    case kAlgIdCBCAES256:
    case kAlgIdGCMAES256:
    {
        keyLen = EVP_CIPHER_key_length(EVP_aes_256_ecb());
    }
        break;
    default:
        break;
    }
    return keyLen/2;
}

netaos::core::Result<SymmetricKey::Uptrc> ImpCryptoProvider::GenerateSymmetricKey(AlgId algId, AllowedUsageFlags allowedUsage, bool isSession, bool isExportable)noexcept{

    CryptoObject::CryptoObjectInfo object_info;
    object_info.objectUid.mCOType = CryptoObjectType::kSymmetricKey;
    auto uuid_res = MakeVersion4Uuid();
    if (!uuid_res) {
        CRYP_ERROR << "Make uuid for symmetric key failed. " << CRYP_ERROR_MESSAGE(uuid_res.Error().Value());
        return netaos::core::Result<SymmetricKey::Uptrc>::FromError(uuid_res.Error());
    }
    object_info.objectUid.mCouid = CryptoObjectUid { uuid_res.Value(), 4 };
    object_info.dependencyUid.mCOType = CryptoObjectType::kSymmetricKey;
    object_info.dependencyUid.mCouid = CryptoObjectUid { Uuid {0, 0}, 4 };
    object_info.isSession = isSession;
    object_info.isExportable = isExportable;

    CryptoPrimitiveId primitive_id(algId);

    std::vector<uint8_t> key(getCipherKeyLength(algId));
    RAND_bytes(key.data(), key.size());
    CRYP_INFO << "GenerateSymmetricKey: keylen " << getCipherKeyLength(algId) <<" algid :" << algId;
    CRYP_INFO << "key: " << CryptoLogger::GetInstance().ToHexString(key.data(), key.size());

    auto uptrc = std::make_unique<const SimplSymmetricKey>(key, object_info, primitive_id, allowedUsage);
    // auto uptrc = std::make_unique<const SymmetricKey>();
    return netaos::core::Result<SymmetricKey::Uptrc>::FromValue(std::move(uptrc));   
}

// ImpCryptoProvider&  ImpCryptoProvider::Instance(){
//     if(!instance_){
//         std::lock_guard<std::mutex> lock(instance_mutex_);
//         instance_ = new ImpCryptoProvider;
//     }
//     return *instance_;
// }

netaos::core::Result<SignerPrivateCtx::Uptr> ImpCryptoProvider::CreateSignerPrivateCtx(AlgId algId) noexcept{
    auto uptr = std::make_unique<ImpSignerPrivateCtx> (algId);
    return netaos::core::Result<SignerPrivateCtx::Uptr>::FromValue(std::move(uptr));   
}

netaos::core::Result<VerifierPublicCtx::Uptr> ImpCryptoProvider::CreateVerifierPublicCtx(AlgId algId) noexcept{
    auto uptr = std::make_unique<ImpVerifierPublicCtx> (algId);
    return netaos::core::Result<VerifierPublicCtx::Uptr>::FromValue(std::move(uptr));   
}


CryptoProvider::AlgId ImpCryptoProvider::ConvertToAlgId(netaos::core::StringView primitiveName) const noexcept {
    CryptoProvider::AlgId ret = kAlgIdUndefined ;
    netaos::core::String name(primitiveName.data());
    for(auto it =toAlgNameMap_.begin();it != toAlgNameMap_.end();it++){
        if(!name.compare(it->second)){
            ret = it->first;
            break;
        }
    }
   return ret;
}

netaos::core::Result<netaos::core::String> ImpCryptoProvider::ConvertToAlgName(AlgId algId) const noexcept {
    netaos::core::String ret;
    auto it = toAlgNameMap_.find(algId);
    if(it != toAlgNameMap_.end()){
        ret = it->second;
    }else{

    }
   return netaos::core::Result<netaos::core::String>::FromValue(std::move(ret));
}

EVP_PKEY* constructPrivateKey(std::string key_data) {
    BIO* bio = BIO_new_mem_buf(key_data.c_str(), key_data.length());

    if (bio == NULL) {
        CRYP_ERROR << "Failed to create a new BIO." ;
        return NULL;
    }

    EVP_PKEY* pkey = PEM_read_bio_PrivateKey(bio, NULL, NULL, NULL);
    if (pkey == NULL) {
        CRYP_ERROR << "Failed to read PEM private key.";
        BIO_free(bio);
        return NULL;
    }

    BIO_free(bio);
    return pkey;
}

netaos::core::Result<PrivateKey::Uptrc> ImpCryptoProvider::LoadPrivateKey(const IOInterface& container) noexcept {
    std::vector<std::uint8_t> payload =  container.GetPayload();
    EVP_PKEY *pkey = nullptr;
    if(!payload.empty()){
        CRYP_INFO << "LoadPrivateKey success";
        std::string keyStr(payload.begin(), payload.end());
        pkey = constructPrivateKey(keyStr);
    } else {
        CRYP_ERROR << "LoadPrivateKey error, pkey is null";
    }

    CryptoPrimitiveId primitive_id(container.GetPrimitiveId());
    CryptoObject::CryptoObjectInfo object_info;
    object_info.objectUid.mCOType = CryptoObjectType::kPrivateKey;
    object_info.objectUid.mCouid = container.GetObjectId();
    object_info.dependencyUid.mCOType = CryptoObjectType::kUndefined;
    object_info.dependencyUid.mCouid = CRYPTO_OBJECT_UID_INITIALIZER;
    object_info.isSession = container.IsObjectSession();
    object_info.isExportable = container.IsObjectExportable();

    AllowedUsageFlags allowed_usage = container.GetAllowedUsage();

    ImpPrivateKey::Uptrc uptrc = std::make_unique<const ImpPrivateKey>(pkey, object_info, primitive_id, allowed_usage);
    return netaos::core::Result<PrivateKey::Uptrc>::FromValue(std::move(uptrc));
    
}

bool ImpCryptoProvider::Init(){
    return true;
}

bool ImpCryptoProvider::Deinit(){
    return true;
}


// netaos::core::Result<PublicKey::Uptrc> LoadPublicKey(const IOInterface& container) noexcept {
    
// }

}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara
