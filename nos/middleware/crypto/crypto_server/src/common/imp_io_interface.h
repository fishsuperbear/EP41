#ifndef ARA_CRYPTO_COMMON_IMP_IO_INTERFACE_H_
#define ARA_CRYPTO_COMMON_IMP_IO_INTERFACE_H_

#include "common/io_interface.h"
#include "common/base_id_types.h"
#include "cryp/cryobj/crypto_object.h"

namespace hozon {
namespace netaos {
namespace crypto {

class ImpIOInterface:public IOInterface {
public:
    struct IOInterfaceInfo {
        CryptoObjectType    objectType;
        CryptoObjectUid     objectUid;
        CryptoAlgId         algId;
        AllowedUsageFlags   usage;
        std::size_t         capacity;
        std::size_t         objectSize;
        bool                isSession;
        bool                isExportable;
        bool                isWritable;
        IOInterfaceInfo()
        : objectType(CryptoObjectType::kUndefined)
        , objectUid()
        , algId(kAlgIdUndefined)
        , usage(kAllowPrototypedOnly)
        , capacity(0)
        , objectSize(0)
        , isSession(true)
        , isExportable(false)
        , isWritable(false) {

        }
    };
    using Uptr = std::unique_ptr<ImpIOInterface>;
    using Uptrc = std::unique_ptr<const ImpIOInterface>;
    // ~IOInterface() noexcept = default;
    ImpIOInterface(IOInterfaceInfo info, std::vector<std::uint8_t> payload) : ioInfo_(info), payload_(payload) {}
    ImpIOInterface(std::size_t capacity_){
        payload_.resize(capacity_);
    };
    ImpIOInterface(const ImpIOInterface& other)
    : ioInfo_(other.ioInfo_)
    , payload_(other.payload_) {

    }
    ImpIOInterface() = default;
    AllowedUsageFlags GetAllowedUsage() const noexcept override{
        return ioInfo_.usage;
    };

    std::size_t GetCapacity() const noexcept override{
        return ioInfo_.capacity;
    };

    CryptoObjectType GetCryptoObjectType() const noexcept override{
        return ioInfo_.objectType;
    };

    CryptoObjectUid GetObjectId() const noexcept override{
        return ioInfo_.objectUid;
    };

    std::size_t GetPayloadSize() const noexcept override{
        return payload_.size();
    };

    CryptoAlgId GetPrimitiveId() const noexcept override{
        return ioInfo_.algId;
    };
    CryptoObjectType GetTypeRestriction() const noexcept override{ //TODO relize
        return ioInfo_.objectType;

    };
    bool IsObjectExportable() const noexcept override{
        return ioInfo_.isExportable;
    };

    bool IsObjectSession() const noexcept override {
        return ioInfo_.isSession;
    };
    bool IsVolatile() const noexcept override{
        return false;
    };
    bool IsValid() const noexcept override{
        return true;
    };

    bool IsWritable() const noexcept override{
        return ioInfo_.isWritable;
    };

    std::vector<std::uint8_t> GetPayload() const noexcept override{
        return payload_;
    };

    void SetPayload(std::vector<std::uint8_t>& payload)noexcept override{
        payload_ = payload;
    };

    // IOInterface& operator=(const IOInterface& other) = default;
    // IOInterface& operator= (IOInterface &&other)=default;

    IOInterfaceInfo ioInfo_;
private:
    std::vector<std::uint8_t> payload_;
    // std::uint8_t *payload_;
    // void *payload_;
};

}  // namespace crypto
}  // namespace netaos
}  // namespace hozon

#endif  // #define ARA_CRYPTO_COMMON_IMP_IO_INTERFACE_H_