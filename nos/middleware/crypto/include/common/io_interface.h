#ifndef ARA_CRYPTO_COMMON_IO_INTERFACE_H_
#define ARA_CRYPTO_COMMON_IO_INTERFACE_H_

#include "crypto_object_uid.h"
#include "base_id_types.h"

namespace hozon {
namespace netaos {
namespace crypto {

class IOInterface {
public:
    using Uptr = std::unique_ptr<IOInterface>;
    using Uptrc = std::unique_ptr<const IOInterface>;
    virtual ~IOInterface() noexcept = default;
    virtual AllowedUsageFlags GetAllowedUsage() const noexcept = 0;
    virtual std::size_t GetCapacity() const noexcept = 0;
    virtual CryptoObjectType GetCryptoObjectType() const noexcept = 0;
    virtual CryptoObjectUid GetObjectId() const noexcept = 0;
    virtual std::size_t GetPayloadSize() const noexcept = 0;
    virtual CryptoAlgId GetPrimitiveId() const noexcept = 0;
    virtual CryptoObjectType GetTypeRestriction() const noexcept = 0;
    virtual bool IsObjectExportable() const noexcept = 0;
    virtual bool IsObjectSession() const noexcept = 0;
    virtual bool IsVolatile() const noexcept = 0;
    virtual bool IsValid() const noexcept = 0;
    virtual bool IsWritable() const noexcept = 0;
    IOInterface& operator=(const IOInterface& other) = default;
    IOInterface& operator= (IOInterface &&other)=default;
    virtual std::vector<std::uint8_t> GetPayload() const noexcept = 0;
    virtual void SetPayload(std::vector<std::uint8_t>& payload)noexcept = 0;
};

}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_COMMON_IO_INTERFACE_H_