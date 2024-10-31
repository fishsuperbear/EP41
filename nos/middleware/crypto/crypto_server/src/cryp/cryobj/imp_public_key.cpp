#include "cryp/cryobj/imp_public_key.h"

#include <memory>
#include <cstdio>
#include <openssl/bio.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include "openssl/bio.h"
#include "common/crypto_logger.hpp"
#include "cryp/cryobj/crypto_object.h"
// #include "cryp/cryobj/imp_crypto_primitive_id.hpp"
#include "common/imp_io_interface.h"
#include <openssl/pem.h>

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

ImpPublicKey::~ImpPublicKey() {
    CRYP_INFO << "ImpPublicKey destructor. add: 0x" << this;
}

bool ImpPublicKey::CheckKey(bool strongCheck) const noexcept{
    return true;
};

netaos::core::Result<netaos::core::Vector<uint8_t> > ImpPublicKey::HashPublicKey(HashFunctionCtx& hashFunc) const noexcept {
    netaos::core::Vector<uint8_t> ret;
    return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromValue(std::move(ret));
};

netaos::core::Result<void> ImpPublicKey::Save(IOInterface& container) const noexcept{
    char* buffer = nullptr;
    int len = 0;
    std::string publicKeyPEM;
    // 创建BIO对象来存储缓冲区
    BIO* bio = BIO_new(BIO_s_mem());
    if (bio) {
        // 将私钥以PEM编码存储到缓冲区中
        if (PEM_write_bio_PUBKEY(bio, const_cast<ImpPublicKey*>(this)->get_pkey())) {
            // 从BIO对象中读取缓冲区内容到字符串中
            len = BIO_get_mem_data(bio, &buffer);
            publicKeyPEM.assign(buffer, len);
        } else {
            CRYP_ERROR << "Failed to write private key to BIO.";
        }
        // 释放BIO对象
        BIO_free(bio);
    } else {
        CRYP_ERROR << "Failed to create BIO.";
    }
    std::vector<uint8_t> payload(publicKeyPEM.begin(), publicKeyPEM.end());
    dynamic_cast<ImpIOInterface&>(container).ioInfo_.usage = const_cast<ImpPublicKey*>(this)->GetAllowedUsage();
    dynamic_cast<ImpIOInterface&>(container).ioInfo_.objectSize = const_cast<ImpPublicKey*>(this)->GetPayloadSize();
    dynamic_cast<ImpIOInterface&>(container).ioInfo_.isSession = const_cast<ImpPublicKey*>(this)->IsSession();
    dynamic_cast<ImpIOInterface&>(container).ioInfo_.objectType = const_cast<ImpPublicKey*>(this)->GetObjectId().mCOType;
    dynamic_cast<ImpIOInterface&>(container).ioInfo_.objectUid = const_cast<ImpPublicKey*>(this)->GetObjectId().mCouid;
    dynamic_cast<ImpIOInterface&>(container).SetPayload(payload);
    return netaos::core::Result<void>();
};


}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara
