#include "common/entry_point.h"

#include "cryp/imp_crypto_provider.h"
#include "x509/imp_x509_provider.h"
#include "keys/imp_key_storage_provider.h"

namespace hozon {
namespace netaos {
namespace crypto {

cryp::ImpCryptoProvider *pimp_crypto_provider = nullptr;
x509::ImpX509Provider *pimp_x509_provider = nullptr;
keys::IMPKeyStorageProvider *pimp_keystorage_provider = nullptr;

cryp::CryptoProvider::Uptr LoadCryptoProvider() noexcept{

    // const CryptoProvider::Uptr loadProvider =  cryp::ImpCryptoProvider::Instance();
    // cryp::CryptoProvider& crypro = cryp::ImpCryptoProvider::Instance();
    // cryp::CryptoProvider* crypro = new cryp::ImpCryptoProvider;
    // if(pimp_crypto_provider){
    //     std::unique_ptr<cryp::ImpCryptoProvider> uptr(pimp_crypto_provider);
    //     return uptr;
    // }else{
        // crypto_provider_ = std::make_unique<cryp::ImpCryptoProvider>();
    // CryptoLogger::GetInstance().CreateLogger("crypto_client");
    // CRYP_INFO << "Crypto_Client Log init finish.";
    pimp_crypto_provider = new (cryp::ImpCryptoProvider);
    std::unique_ptr<cryp::ImpCryptoProvider> uptr(pimp_crypto_provider);
    return uptr;
    // }

    // return std::make_unique<cryp::ImpCryptoProvider>();
    //  cryp::CryptoProvider::Uptr uptr(crypro);
    //  return uptr;
};

x509::X509Provider::Uptr LoadX509Provider() noexcept{
    // return std::make_unique<x509::ImpX509Provider>();
    // if (pimp_x509_provider) {
    //     std::unique_ptr<x509::ImpX509Provider> uptr(pimp_x509_provider);
    //     return uptr;
    // } else {
        pimp_x509_provider = new (x509::ImpX509Provider);
        std::unique_ptr<x509::ImpX509Provider> uptr(pimp_x509_provider);
        return uptr;
    // }
};

keys::KeyStorageProvider::Uptr LoadKeyStorageProvider() noexcept{
    // return std::make_unique<keys::IMPKeyStorageProvider>();
    // if (pimp_keystorage_provider) {
    //     std::unique_ptr<keys::IMPKeyStorageProvider> uptr(pimp_keystorage_provider);
    //     return uptr;
    // } else {
        pimp_keystorage_provider = new (keys::IMPKeyStorageProvider);
        std::unique_ptr<keys::IMPKeyStorageProvider> uptr(pimp_keystorage_provider);
        return uptr;
    // }
};


}  // namespace crypto
}  // namespace ara
}  // namespace ara
