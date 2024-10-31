#ifndef ARA_CRYPTO_CRYP_EXTENSION_SERVICE_H_
#define ARA_CRYPTO_CRYP_EXTENSION_SERVICE_H_

#include "common/base_id_types.h"
#include "common/crypto_object_uid.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class ExtensionService{
public:
    using Uptr = std::unique_ptr<ExtensionService>;

    struct ExtensionServiceInfo {
        uint64_t actual_key_bit_length;
        CryptoObjectUid actual_key_couid;
        AllowedUsageFlags allowed_usage;
        uint64_t max_key_bit_length;
        uint64_t min_key_bit_length;
        bool is_key_bit_length_supported;
        bool is_key_available;

        ExtensionServiceInfo()
        : actual_key_bit_length(0)
        , actual_key_couid()
        , allowed_usage(kAllowPrototypedOnly)
        , max_key_bit_length(0)
        , min_key_bit_length(0)
        , is_key_bit_length_supported(false)
        , is_key_available(false) {

        }

        ExtensionServiceInfo(uint64_t _actual_key_bit_length
            , CryptoObjectUid _actual_key_couid
            , AllowedUsageFlags _allowed_usage
            , uint64_t _max_key_bit_length
            , uint64_t _min_key_bit_length
            , bool _is_key_bit_length_supported
            , bool _is_key_available)
        : actual_key_bit_length(_actual_key_bit_length)
        , actual_key_couid(_actual_key_couid)
        , allowed_usage(_allowed_usage)
        , max_key_bit_length(_max_key_bit_length)
        , min_key_bit_length(_min_key_bit_length)
        , is_key_bit_length_supported(_is_key_bit_length_supported)
        , is_key_available(_is_key_available) {

        }

        ExtensionServiceInfo(const ExtensionServiceInfo& other)
        : actual_key_bit_length(other.actual_key_bit_length)
        , actual_key_couid(other.actual_key_couid)
        , allowed_usage(other.allowed_usage)
        , max_key_bit_length(other.max_key_bit_length)
        , min_key_bit_length(other.min_key_bit_length)
        , is_key_bit_length_supported(other.is_key_bit_length_supported)
        , is_key_available(other.is_key_available) {

        }

        ExtensionServiceInfo& operator= (const ExtensionServiceInfo& other) {
            actual_key_bit_length = other.actual_key_bit_length;
            actual_key_couid = other.actual_key_couid;
            allowed_usage = other.allowed_usage;
            max_key_bit_length = other.max_key_bit_length;
            min_key_bit_length = other.min_key_bit_length;
            is_key_bit_length_supported = other.is_key_bit_length_supported;
            is_key_available = other.is_key_available;
            return *this;
        }
    };

    virtual ~ExtensionService () noexcept=default;
    virtual std::size_t GetActualKeyBitLength () const noexcept=0;
    virtual CryptoObjectUid GetActualKeyCOUID () const noexcept=0;
    virtual AllowedUsageFlags GetAllowedUsage () const noexcept=0;
    virtual std::size_t GetMaxKeyBitLength () const noexcept=0;
    virtual std::size_t GetMinKeyBitLength () const noexcept=0;
    virtual bool IsKeyBitLengthSupported(std::size_t keyBitLength) const noexcept = 0;
    virtual bool IsKeyAvailable () const noexcept=0;
    ExtensionService& operator=(const ExtensionService& other) = default;
    ExtensionService& operator=(ExtensionService&& other) = default;

   private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_EXTENSION_SERVICE_H_