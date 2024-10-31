#include "cryp/imp_hash_function_ctx.h"

#include <iostream>
#include <cstddef>

#include "common/imp_volatile_trusted_container.h"
#include "common/crypto_logger.hpp"
#include "cryp/imp_crypto_provider.h"
// #include "cryp/cryobj/imp_crypto_primitive_id.hpp"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

netaos::core::Result<void> ImpHashFunctionCtx::Start() noexcept{
    std::string algorithmName;
    openssl_ctx_.plibrary_context = OSSL_LIB_CTX_new();
    if (!openssl_ctx_.plibrary_context) {
        CRYP_ERROR<<"OSSL_LIB_CTX_new() returned NULL.";
    }

    // Fetch a message digest by name.The algorithm name is case insensitive. 
    switch (alg_id_)
    {
    case kAlgIdSHA256:
        algorithmName = "SHA2-256";
        break;
    case kAlgIdSHA384:
        algorithmName = "SHA2-384";
        break;
    case kAlgIdSHA512:
        algorithmName = "SHA2-512";
        break;
    default:
        break;
    }

    openssl_ctx_.pmessage_digest = EVP_MD_fetch(openssl_ctx_.plibrary_context,static_cast<const char *>(algorithmName.data()), openssl_ctx_.poption_properties);
    if (!openssl_ctx_.pmessage_digest) {
        CRYP_ERROR<< "EVP_MD_fetch could not find SHA3-512.";
    }
    //Determine the length of the fetched digest type 
    openssl_ctx_.digest_length = EVP_MD_get_size(openssl_ctx_.pmessage_digest);
    if (openssl_ctx_.digest_length <= 0) {
        CRYP_ERROR<<"EVP_MD_get_size returned invalid size.";
    }

    openssl_ctx_.pdigest_value = (unsigned char*)OPENSSL_malloc(openssl_ctx_.digest_length);
    if (!openssl_ctx_.pdigest_value) {
        CRYP_ERROR<< "No memory.";
    }

    // Make a message digest context to hold temporary state during digest creation
    openssl_ctx_.pdigest_context = EVP_MD_CTX_new();
    if (!openssl_ctx_.pdigest_context) {
        CRYP_ERROR<<"EVP_MD_CTX_new failed.";
    }

    //Initialize the message digest context to use the fetched digest provider
    if (EVP_DigestInit(openssl_ctx_.pdigest_context, openssl_ctx_.pmessage_digest) != 1) {
        CRYP_ERROR<<"EVP_DigestInit failed.";
    }

    return netaos::core::Result<void>();

}

netaos::core::Result<void> ImpHashFunctionCtx::Update(std::vector<uint8_t>& in) noexcept{
    input_.assign(in.begin(),in.end());
    // input_.resize(in.size());
    if (EVP_DigestUpdate(openssl_ctx_.pdigest_context, input_.data(), input_.size()) != 1) {
        CRYP_ERROR<<"EVP_DigestUpdate(in) failed.";
    }
    return netaos::core::Result<void>();
}

netaos::core::Result<netaos::core::Vector<uint8_t>> ImpHashFunctionCtx::Finish() noexcept{
    std::vector<uint8_t> out;
    if (EVP_DigestFinal(openssl_ctx_.pdigest_context, openssl_ctx_.pdigest_value, &openssl_ctx_.digest_length) != 1) {
        CRYP_ERROR<<"EVP_DigestFinal() failed.";
    }
    std::vector<std::uint8_t> data(openssl_ctx_.pdigest_value,openssl_ctx_.pdigest_value+openssl_ctx_.digest_length);
    if(openssl_ctx_.pdigest_value){
        out.resize(openssl_ctx_.digest_length);
        for(uint32_t i=0;i<openssl_ctx_.digest_length;i++){
            CRYP_INFO<<static_cast<uint32_t>(data[i])<<" ";
            out[i] = static_cast<uint8_t>(openssl_ctx_.pdigest_value[i]);
        }
    }
    // for(uint32_t i=0;i<out.size();i++){
    //     // CRYP_INFO<<static_cast<uint8_t>(out[i])<<" ";
    //         CRYP_INFO<<std::hex<<static_cast<uint32_t>(out[i])<<" ";

    // }
    CRYP_INFO<<"";
    return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromValue(std::move(out));
}

// void ImpHashFunctionCtx::FreeOpensslCtx() noexcept{

// }

netaos::core::Result<std::vector<uint8_t>> ImpHashFunctionCtx::GetDigest(std::size_t offset) const noexcept {
    std::vector<uint8_t> out;
    for (uint32_t i = 0; i < openssl_ctx_.digest_length; i++) {
        out[i] = static_cast<uint8_t>(openssl_ctx_.pdigest_value[i]);
    }
    return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromValue(std::move(out));
}

bool ImpHashFunctionCtx::IsInitialized() const noexcept{
    return true;
}

CryptoPrimitiveId::Uptr ImpHashFunctionCtx::GetCryptoPrimitiveId() const noexcept {
    return CryptoPrimitiveId::Uptr(new CryptoPrimitiveId(alg_id_));
}

// ImpHashFunctionCtx::ImpHashFunctionCtx(){
//     input_ = {}
// }

CryptoProvider& ImpHashFunctionCtx::MyProvider() const noexcept{
    // return const_cast<ImpCryptoProvider&>(ImpCryptoProvider::Instance());
    // CryptoProvider& prov = ImpCryptoProvider::Instance();
    CryptoProvider* prov = new ImpCryptoProvider;
    return *prov;
}

}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara
