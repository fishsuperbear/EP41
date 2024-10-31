#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <map>
#include "common/io_interface.h"
#include "cryp/cryobj/symmetric_key.h"
#include "cryp/cryobj/private_key.h"
#include "cryp/cryobj/public_key.h"
#include "cryp/symmetric_block_cipher_ctx.h"
#include "cryp/encryptor_public_ctx.h"
#include "cryp/decryptor_private_ctx.h"
#include "cryp/signer_private_ctx.h"
#include "cryp/verifier_public_ctx.h"
#include "keys/keyslot.h"
#include "x509/cert_sign_request.h"

namespace hozon {
namespace netaos {
namespace crypto {

class ResourceKeeper {

public:
    static ResourceKeeper& Instance();
    static void Destroy();

    bool Init();
    void Deinit();

    // template < typename T>
    // void* Query(uint64_t ref);

    // template < typename T = cryp::SymmetricKey>
    cryp::SymmetricKey* QuerySymmetricKey(uint64_t ref);
    // template < typename T = cryp::PrivateKey>
    cryp::PrivateKey* QueryPrivateKey(uint64_t ref);
    // template < typename T = cryp::PublicKey>
    cryp::PublicKey* QueryPublicKey(uint64_t ref);

    // template < typename T = cryp::SymmetricBlockCipherCtx>
    cryp::SymmetricBlockCipherCtx* QuerySymmetricBlockCipherCtx(uint64_t ref);

    cryp::EncryptorPublicCtx* QueryEncryptorPublicCtx(uint64_t ref);

    cryp::DecryptorPrivateCtx* QueryDecryptorPrivateCtx(uint64_t ref);

    cryp::SignerPrivateCtx* QuerySignerPrivateCtx(uint64_t ref);

    cryp::VerifierPublicCtx* QueryVerifierPublicCtx(uint64_t ref);
    IOInterface* QueryIoInterfaceContainer(uint64_t ref);

    keys::KeySlot* QueryKeySlot(uint64_t ref);
    x509::CertSignRequest* QueryCertSignRequest(uint64_t ref);

    x509::X509DN* QueryX509Dn(uint64_t ref);

    // template < typename T>
    // uint64_t Keep(T* res);

    // template < >
    uint64_t KeepSymmetricKey(cryp::SymmetricKey* res);
    // template < typename T = cryp::PrivateKey>
    uint64_t KeepPrivateKey(cryp::PrivateKey* res);
    // template < typename T = cryp::PublicKey>
    uint64_t KeepPublicKey(cryp::PublicKey* res);

    // template < typename T = cryp::SymmetricBlockCipherCtx>
    uint64_t KeepSymmetricBlockCipherCtx(cryp::SymmetricBlockCipherCtx* res);

    uint64_t KeepEncryptorPublicCtx(cryp::EncryptorPublicCtx* res);

    uint64_t KeepDecryptorPrivateCtx(cryp::DecryptorPrivateCtx* res);

    uint64_t KeepSignerPrivateCtx(cryp::SignerPrivateCtx* res);

    uint64_t KeepVerifierPublicCtx(cryp::VerifierPublicCtx* res);
    uint64_t KeepIoInterfaceContainer(IOInterface* res);

    uint64_t KeepKeySlot(keys::KeySlot* res);
    uint64_t KeepCertSignRequest(x509::CertSignRequest* res);

    uint64_t KeepX509DN(x509::X509DN* res);

    void Release(uint64_t ref);

    enum ResourceType : uint64_t {
        kResourceTypeSymmetricKey = 0,
        kResourceTypePrivateKey,
        kResourceTypePublicKey,
        kResourceTypeSymmetricBlockCipherCtx = 10,
        kResourceTypeEncryptorPublicCtx,
        kResourceTypeDecryptorPrivateCtx,
        kResourceTypeSignerPrivateCtx,
        kResourceTypeVerifierPublicCtx,
        kResourceTypeIoInterfaceContainer,
        kResourceTypeKeySlot,
        kResourceTypeCertSignRequest,
        kResourceTypeX509DnRequest
    };

private:
    ResourceKeeper();
    ~ResourceKeeper();

    uint64_t KeepResource(int32_t type, void* res);

    struct ResourceInfo {
        int32_t type;
        void* res;
    };
    
    std::map<uint64_t, ResourceInfo> resource_map_;
    std::recursive_mutex resource_mutex_;
};

}
}
}