
#include "cryp/imp_symmetric_block_cipher_ctx.h"

#include <memory>

#include "cryp/imp_crypto_provider.h"
#include "cryp/imp_crypto_service.h"
#include "common/crypto_error_domain.h"
#include "common/crypto_logger.hpp"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

// static const unsigned char aes_key[] = "0123456789abcdeF";

ImpSymmetricBlockCipherCtx::ImpSymmetricBlockCipherCtx(CryptoAlgId alg_id)
: alg_id_(alg_id)
, symmetric_key_(nullptr) {

}

CryptoService::Uptr ImpSymmetricBlockCipherCtx::GetCryptoService() const noexcept{
    return std::make_unique<ImpCryptoService>();
}

netaos::core::Result<CryptoTransform> ImpSymmetricBlockCipherCtx::GetTransformation() const noexcept {
    return netaos::core::Result<CryptoTransform>::FromValue(transform_);
}

netaos::core::Result<netaos::core::Vector<uint8_t>> ImpSymmetricBlockCipherCtx::ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding) const noexcept {
    int inlength = in.size();
    const unsigned char *pindata = in.data();
    unsigned char *poutdata = const_cast<unsigned char*>(out_.data());
    int outtemplen = 0;
    if(suppressPadding){
        EVP_CIPHER_CTX_set_padding(ctx_, 1);
    }
    CRYP_INFO << "ProcessBlock ctx_ addr:"<<ctx_<<" inlength:"<<inlength;
    if(inlength > 1024){
        while (inlength > 1024) {
            if (!EVP_CipherUpdate(ctx_, poutdata, &outtemplen, pindata, inlength)) {
                CRYP_ERROR << "EVP_CipherUpdate error.";
            }
            pindata += 1024;
            inlength -= 1024;
            poutdata += outtemplen;
            outlen_ += outtemplen;
        }
    }else{
        if (!EVP_CipherUpdate(ctx_, poutdata, &outtemplen, pindata, inlength)) {
            CRYP_ERROR << "EVP_CipherUpdate error.";
        }else{
            CRYP_INFO << "EVP_CipherUpdate finish.";
        }
        poutdata += outtemplen;
        outlen_ += outtemplen;
    }
   
    CRYP_INFO << "outtemplen:"<<outtemplen;
    if (!EVP_CipherFinal_ex(ctx_, poutdata, &outtemplen)) {
        CRYP_ERROR<< "EVP_CipherFinal error.";
    }
    outlen_ += outtemplen;
    CRYP_INFO<<"outlen_:" <<outlen_;
    netaos::core::Vector<uint8_t> out;
    out.resize(outlen_);
    CRYP_INFO<<"ImpSymmetricBlockCipherCtx::ProcessBlock encrypted data: ";
    for(int i=0;i<outlen_;i++){
        CRYP_INFO<<static_cast<uint32_t>(out_[i]) <<" ";
        out[i] = static_cast<uint8_t>(out_[i]);
    }
    EVP_CIPHER_CTX_free(ctx_);

    return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromValue(std::move(out));
}

netaos::core::Result<netaos::core::Vector<uint8_t>> ImpSymmetricBlockCipherCtx::ProcessBlocks(ReadOnlyMemRegion in) const noexcept{
    netaos::core::Vector<uint8_t> out;
    return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromValue(std::move(out));

}

netaos::core::Result<void> ImpSymmetricBlockCipherCtx::Reset() noexcept{
    return netaos::core::Result<void>();
}

const EVP_CIPHER *ImpSymmetricBlockCipherCtx::getEvpCipher() {
    const EVP_CIPHER * cipher = nullptr;
    switch (alg_id_)
    {
    case kAlgIdCBCAES128:
    {
        cipher = EVP_aes_128_cbc();
    }
        break;
    case kAlgIdECBAES128:
    {
        cipher = EVP_aes_128_ecb();
    }
        break;
    case kAlgIdGCMAES128:
    {
        cipher = EVP_aes_128_gcm();
    }
        break;
    case kAlgIdCBCAES192:
    {
        cipher = EVP_aes_128_cbc();
    }
        break;
    case kAlgIdGCMAES192:
    {
        cipher = EVP_aes_192_gcm();
    }
        break;
    case kAlgIdCBCAES256:
    {
        cipher = EVP_aes_128_cbc();
    }
        break;
    case kAlgIdGCMAES256:
    {
        cipher = EVP_aes_256_gcm();
    }
        break;
    default:
        break;
    }
    return cipher;
}

netaos::core::Result<void> ImpSymmetricBlockCipherCtx::SetKey(const SymmetricKey& key, CryptoTransform transform) noexcept{
    symmetric_key_.reset(new SimplSymmetricKey(dynamic_cast<const SimplSymmetricKey&>(key)));
    
    int blockSize = 0;
    transform_ = transform;
    // symmetricKey_ = key;
    unsigned char iv[] = "1234567887654321";
 
    ctx_ = EVP_CIPHER_CTX_new();
    EVP_CIPHER_CTX_init(ctx_);
    // cipher_ = EVP_get_cipherbyname("aes-128-cbc");
    cipher_ = getEvpCipher();

    if(transform == CryptoTransform::kEncrypt){
        // EVP_CipherInit_ex(ctx_, cipher_, NULL, aes_key,iv,1);
        // TODO: comment out before 1220.
        CRYP_INFO << "ImpSymmetricBlockCipherCtx SetKey : "
            << CryptoLogger::GetInstance().ToHexString(symmetric_key_->GetKeyData().data(), symmetric_key_->GetKeyData().size());

        if (!EVP_CipherInit_ex2(ctx_, cipher_, symmetric_key_->GetKeyData().data(), iv, 1, NULL)) {
            EVP_CIPHER_CTX_free(ctx_);
            //  return netaos::core::Result<void>::FromError(netaos::crypto::CryptoErrc::kResourceFault);
            // return 0;
         }
    }else if(transform == CryptoTransform::kDecrypt){
        // EVP_CipherInit_ex(ctx_, cipher_, NULL, aes_key,iv,0);
        // TODO
        if (!EVP_CipherInit_ex2(ctx_, cipher_, symmetric_key_->GetKeyData().data(), iv, 0, NULL)) {
             EVP_CIPHER_CTX_free(ctx_);
             // return 0;
        }
    }else{

    }
    
    blockSize = EVP_CIPHER_get_block_size(getEvpCipher());
    CRYP_INFO<<"SetKey bock_size:"<<blockSize;
    out_.resize(1024+blockSize);
    outlen_ = 0;
    isInitialized_ = true;

    return netaos::core::Result<void>();
}

CryptoPrimitiveId::Uptr ImpSymmetricBlockCipherCtx::GetCryptoPrimitiveId () const noexcept {
    return CryptoPrimitiveId::Uptr(new CryptoPrimitiveId(alg_id_));
}

bool ImpSymmetricBlockCipherCtx::IsInitialized() const noexcept{
    return isInitialized_;
}

CryptoProvider& ImpSymmetricBlockCipherCtx::MyProvider() const noexcept{
    CryptoProvider* prov = new ImpCryptoProvider;
    return *prov;
}

// ImpSymmetricBlockCipherCtx::ImpSymmetricBlockCipherCtx(){

// }

}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara
