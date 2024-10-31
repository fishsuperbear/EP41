#include "common/entry_point.h"

#include "cryp/cimpl_crypto_provider.h"
#include "x509/cimp_x509_provider.h"
#include "keys/cimpl_key_storage_provider.h"
#include "common/crypto_logger.hpp"

namespace hozon {
namespace netaos {
namespace crypto {

cryp::CimplCryptoProvider *pimp_crypto_provider = nullptr;
keys::CimplKeyStorageProvider *pimp_keyStorage_provider = nullptr;
x509::CimpX509Provider *pimp_x509_provider = nullptr;

cryp::CryptoProvider::Uptr LoadCryptoProvider() noexcept{
    CryptoLogger::GetInstance().CreateLogger("crypto_client");
    CRYP_INFO<< "crypto_client init finish.";
    pimp_crypto_provider = new(cryp::CimplCryptoProvider);
    std::unique_ptr<cryp::CimplCryptoProvider> uptr(pimp_crypto_provider);
    return uptr;
};

x509::X509Provider::Uptr LoadX509Provider() noexcept{
    pimp_x509_provider = new(x509::CimpX509Provider);
    std::unique_ptr<x509::CimpX509Provider> uptr(pimp_x509_provider);
    return uptr;
};

keys::KeyStorageProvider::Uptr LoadKeyStorageProvider() noexcept{
    pimp_keyStorage_provider = new(keys::CimplKeyStorageProvider);
    std::unique_ptr<keys::CimplKeyStorageProvider> uptr(pimp_keyStorage_provider);
    return uptr;
};


}  // namespace crypto
}  // namespace ara
}  // namespace ara
