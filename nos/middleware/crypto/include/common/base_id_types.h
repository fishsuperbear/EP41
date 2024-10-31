#ifndef ARA_CRYPTO_COMMON_BASE_ID_TYPES_H_
#define ARA_CRYPTO_COMMON_BASE_ID_TYPES_H_
#include <cstdint>
#include <memory>
#include <vector>
#include <map>
#include <string>

#include "core/map.h"
#include "core/span.h"
#include "core/vector.h"
#include "core/string.h"
#include "core/string_view.h"
#include "core/result.h"
#include "core/error_code.h"
#include "core/error_domain.h"
#include "core/exception.h"

namespace hozon {
namespace netaos {
namespace crypto {
using namespace netaos;
using AllowedUsageFlags = std::uint32_t;
using DefBytesAllocator = std::allocator<std::uint8_t>;
template < typename Alloc = DefBytesAllocator>
using ByteVector = netaos::core::Vector<std::uint8_t,Alloc>;
// using ByteVector = std::vector<std::uint8_t,Alloc>;
using CryptoAlgId = std::uint64_t;

enum class CryptoObjectType : std::uint32_t {
    kUndefined = 0,
    kSymmetricKey = 1,
    kPrivateKey = 2,
    kPublicKey = 3,
    kSignature= 4,
    kSecretSeed= 5
};

enum class ProviderType : std::uint32_t {
    kUndefinedProvider= 0,
    kCryptoProvider = 1,
    kKeyStorageProvider = 2,
    kX509Provider = 3
};

enum class CryptoTransform : std::uint32_t {
    kTransformNotSet = 0,
    kEncrypt = 1,
    kDecrypt = 2,
    kMacVerify = 3,
    kMacGenerate = 4,
    kWrap = 5,
    kUnwrap = 6,
    kSigVerify = 7,
    kSigGenerate = 8
};

enum class KeySlotType : std::uint32_t {
    kUnknown = 0,
    kMachine = 1,
    kApplication = 2
};


constexpr CryptoAlgId kAlgIdUndefined = 0u;
constexpr CryptoAlgId kAlgIdAny = kAlgIdUndefined;
constexpr CryptoAlgId kAlgIdDefault = kAlgIdUndefined;
constexpr CryptoAlgId kAlgIdNone = kAlgIdUndefined;

constexpr CryptoAlgId kAlgIdCBCAES128 = 0X0001ULL;
constexpr CryptoAlgId kAlgIdCBCAES192 = 0X0002ULL;
constexpr CryptoAlgId kAlgIdCBCAES256 = 0X0004ULL;

constexpr CryptoAlgId kAlgIdECBAES128 =  0X0060ULL;

constexpr CryptoAlgId kAlgIdGCMAES128 = 0X0008ULL;
constexpr CryptoAlgId kAlgIdGCMAES192 = 0X0010ULL;
constexpr CryptoAlgId kAlgIdGCMAES256 = 0X0020ULL;

constexpr CryptoAlgId kAlgIdTRNG =  0X0040ULL;

constexpr CryptoAlgId kAlgIdMD5    = 0X0070ULL;
constexpr CryptoAlgId kAlgIdSHA1   = 0X0080ULL;
constexpr CryptoAlgId kAlgIdSHA256 = 0X0100ULL;
constexpr CryptoAlgId kAlgIdSHA384 = 0X0200ULL;
constexpr CryptoAlgId kAlgIdSHA512 = 0X0400ULL;

constexpr CryptoAlgId kAlgIdHMACSHA256 = 0X0800ULL;

constexpr CryptoAlgId kAlgIdCMACAES128 = 0X4000ULL;
constexpr CryptoAlgId kAlgIdCMACAES256 = kAlgIdCMACAES128 << 1U;

constexpr CryptoAlgId kAlgIdCBCSM4  = 0X8001ULL;
constexpr CryptoAlgId kAlgIdGCMSM4  = 0X8002ULL;
constexpr CryptoAlgId kAlgIdCTRSM4  = 0X8003ULL;
constexpr CryptoAlgId kAlgIdECBSM4  = 0X8004ULL;
constexpr CryptoAlgId kAlgIdSM2SIGN = 0X8005ULL;
constexpr CryptoAlgId kAlgIdSM3     = 0X8006ULL;

constexpr CryptoAlgId kAlgIdRSA2048SHA384PSS   = 0X8007ULL;
constexpr CryptoAlgId kAlgIdRSA2048SHA512PSS   = 0X8008ULL;
constexpr CryptoAlgId kAlgIdRSA3072SHA256PSS   = 0X8009ULL;
constexpr CryptoAlgId kAlgIdRSA3072SHA512PSS   = 0X800AULL;
constexpr CryptoAlgId kAlgIdRSA4096SHA256PSS   = 0X800BULL;
constexpr CryptoAlgId kAlgIdRSA4096SHA384PSS   = 0X800CULL;
constexpr CryptoAlgId kAlgIdRSA2048SHA384PKCS  = 0X800DULL;
constexpr CryptoAlgId kAlgIdRSA2048SHA512PKCS  = 0X800EULL;
constexpr CryptoAlgId kAlgIdRSA3072SHA256PKCS  = 0X800FULL;
constexpr CryptoAlgId kAlgIdRSA3072SHA512PKCS  = 0X8010ULL;
constexpr CryptoAlgId kAlgIdRSA4096SHA256PKCS  = 0X8011ULL;
constexpr CryptoAlgId kAlgIdRSA4096SHA384PKCS  = 0X8012ULL;

constexpr CryptoAlgId kAlgIdRSA2048SHA256PSS   = kAlgIdCMACAES256            << 1U;
constexpr CryptoAlgId kAlgIdRSA3072SHA384PSS   = kAlgIdRSA2048SHA256PSS      << 1U;
constexpr CryptoAlgId kAlgIdRSA4096SHA512PSS   = kAlgIdRSA3072SHA384PSS      << 1U;
constexpr CryptoAlgId kAlgIdRSA2048SHA256PKCS  = kAlgIdRSA4096SHA512PSS      << 1U;
constexpr CryptoAlgId kAlgIdRSA3072SHA384PKCS  = kAlgIdRSA2048SHA256PKCS     << 1U;
constexpr CryptoAlgId kAlgIdRSA4096SHA512PKCS  = kAlgIdRSA3072SHA384PKCS     << 1U;
constexpr CryptoAlgId kAlgIdED25519SHA256      = kAlgIdRSA4096SHA512PKCS     << 1U;
constexpr CryptoAlgId kAlgIdECDSASHA256        = kAlgIdED25519SHA256         << 1U;
constexpr CryptoAlgId kAlgIdECDSASHA384        = kAlgIdECDSASHA256           << 1U;
constexpr CryptoAlgId kAlgIdECDSASHA512        = kAlgIdECDSASHA384           << 1U;

constexpr CryptoAlgId kAlgIdRSA2048SHA256OAEP  = kAlgIdECDSASHA512           << 1U;
constexpr CryptoAlgId kAlgIdRSA2048SHA384OAEP  = kAlgIdRSA2048SHA256OAEP     << 1U;
constexpr CryptoAlgId kAlgIdRSA2048SHA512OAEP  = kAlgIdRSA2048SHA384OAEP     << 1U;

constexpr CryptoAlgId kAlgIdRSA4096SHA256OAEP  = kAlgIdRSA2048SHA512OAEP     << 1U;
constexpr CryptoAlgId kAlgIdRSA4096SHA384OAEP  = kAlgIdRSA4096SHA256OAEP     << 1U;
constexpr CryptoAlgId kAlgIdRSA4096SHA512OAEP  = kAlgIdRSA4096SHA384OAEP     << 1U;

constexpr CryptoAlgId kAlgIdPBKDF2             = kAlgIdRSA4096SHA512OAEP     << 1U;

const AllowedUsageFlags kAllowPrototypedOnly = 0;
const AllowedUsageFlags kAllowDataEncryption = 0x0001;
const AllowedUsageFlags kAllowDataDecryption = 0x0002;
const AllowedUsageFlags kAllowSignature = 0x0004;
const AllowedUsageFlags kAllowVerification = 0x0008;
const AllowedUsageFlags kAllowKeyAgreement = 0x0010;
const AllowedUsageFlags kAllowKeyDiversify = 0x0020;
const AllowedUsageFlags kAllowRngInit = 0x0040;
const AllowedUsageFlags kAllowKdfMaterial = 0x0080;
const AllowedUsageFlags kAllowKeyExporting = 0x0100;
const AllowedUsageFlags kAllowKeyImporting = 0x0200;
const AllowedUsageFlags kAllowExactModeOnly = 0x8000;
const AllowedUsageFlags kAllowDerivedDataDecryption = kAllowDataDecryption << 16;
const AllowedUsageFlags kAllowDerivedDataEncryption = kAllowDataEncryption << 16;
const AllowedUsageFlags kAllowDerivedRngInit = kAllowRngInit << 16;
const AllowedUsageFlags kAllowDerivedExactModeOnly = kAllowExactModeOnly << 16;
const AllowedUsageFlags kAllowDerivedKdfMaterial = kAllowKdfMaterial << 16;
const AllowedUsageFlags kAllowDerivedKeyDiversify = kAllowKeyDiversify << 16;
const AllowedUsageFlags kAllowDerivedKeyExporting = kAllowKeyExporting << 16;
const AllowedUsageFlags kAllowDerivedKeyImporting = kAllowKeyImporting << 16;
const AllowedUsageFlags kAllowDerivedSignature = kAllowSignature << 16;
const AllowedUsageFlags kAllowDerivedVerification = kAllowVerification << 16;
const AllowedUsageFlags kAllowKdfMaterialAnyUsage = kAllowKdfMaterial | kAllowDerivedDataEncryption | kAllowDerivedDataDecryption | kAllowDerivedSignature | kAllowDerivedVerification |
                                                    kAllowDerivedKeyDiversify | kAllowDerivedRngInit | kAllowDerivedKdfMaterial | kAllowDerivedKeyExporting | kAllowDerivedKeyImporting;

const netaos::core::String ALG_NAME_CBC_AES128 = "aes-128-cbc";
const netaos::core::String ALG_NAME_CBC_AES192 = "aes-192-cbc";
const netaos::core::String ALG_NAME_CBC_AES256 = "aes-256-cbc";

const netaos::core::String ALG_NAME_GCM_AES128 = "aes-128-gcm";
const netaos::core::String ALG_NAME_GCM_AES192 = "aes-192-gcm";
const netaos::core::String ALG_NAME_GCM_AES256 = "aes-256-gcm";

const netaos::core::String ALG_NAME_ECB_AES128  = "aes-128-ecb";

const netaos::core::String ALG_NAME_MD5  = "dgst-md5";
const netaos::core::String ALG_NAME_SHA1 = "dgst-sha1";
const netaos::core::String ALG_NAME_SHA256 = "dgst-sha256";
const netaos::core::String ALG_NAME_SHA384 = "dgst-sha384";
const netaos::core::String ALG_NAME_SHA512 = "dgst-sha512";

}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_COMMON_BASE_ID_TYPES_H_