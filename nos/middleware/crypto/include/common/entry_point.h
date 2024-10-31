#ifndef ARA_CRYPTO_COMMON_ENTRY_POINT_H_
#define ARA_CRYPTO_COMMON_ENTRY_POINT_H_
#include <cstdint>
#include <memory>
#include <vector>
// #include "elementary_types.h"
#include "cryp/crypto_provider.h"
#include "x509/x509_provider.h"
#include "keys/key_storage_provider.h"

namespace hozon {
namespace netaos {
namespace crypto {

struct SecureCounter {
    std::uint64_t mLSQW;
    std::uint64_t mMSQW;
};

cryp::CryptoProvider::Uptr LoadCryptoProvider() noexcept;
x509::X509Provider::Uptr LoadX509Provider() noexcept;
keys::KeyStorageProvider::Uptr LoadKeyStorageProvider() noexcept;
// cryp::CryptoProvider::Uptr LoadCryptoProvider(const ara::core::InstanceSpecifier& iSpecify) noexcept;
// keys::KeyStorageProvider::Uptr LoadKeyStorageProvider () noexcept;
// x509::X509Provider::Uptr LoadX509Provider () noexcept;

// ara::core::Result<ara::core::Vector<ara::core::Byte> > GenerateRandomData(std::uint32_t count) noexcept;
// ara::core::Result<SecureCounter> GetSecureCounter () noexcept;
std::vector<uint8_t> GenerateRandomData(std::uint32_t count) noexcept;
SecureCounter& GetSecureCounter() noexcept;




}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_COMMON_ENTRY_POINT_H_