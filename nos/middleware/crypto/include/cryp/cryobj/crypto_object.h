#ifndef ARA_CRYPTO_CRYP_CRYPTO_OBJECT_H_
#define ARA_CRYPTO_CRYP_CRYPTO_OBJECT_H_
#include "core/result.h"
#include "common/base_id_types.h"
#include "common/crypto_object_uid.h"
#include "common/io_interface.h"
#include "common/crypto_error_domain.h"
#include "cryp/cryobj/crypto_primitive_id.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class CryptoObject {
public:
    struct COIdentifier {
        CryptoObjectType mCOType;
        CryptoObjectUid mCouid;

        COIdentifier()
        : mCOType(CryptoObjectType::kUndefined)
        , mCouid() {

        }

        COIdentifier(const COIdentifier& other)
        : mCOType(other.mCOType)
        , mCouid(other.mCouid) {
            
        }

        COIdentifier& operator= (const COIdentifier& other) {
            mCOType = other.mCOType;
            mCouid = other.mCouid;
            return *this;
        }

    };
    struct CryptoObjectInfo {
        COIdentifier      objectUid;
        COIdentifier      dependencyUid;
        bool              isSession;
        bool              isExportable;
        uint64_t          payloadSize;

        CryptoObjectInfo()
        : objectUid()
        , dependencyUid()
        , isSession(true)
        , isExportable(false)
        , payloadSize(0) {

        }

        CryptoObjectInfo(const CryptoObjectInfo& other)
        : objectUid(other.objectUid)
        , dependencyUid(other.dependencyUid)
        , isSession(other.isSession)
        , isExportable(other.isExportable)
        , payloadSize(other.payloadSize) {

        }

        CryptoObjectInfo& operator= (const CryptoObjectInfo& other) {
            objectUid = other.objectUid;
            dependencyUid = other.dependencyUid;
            isSession = other.isSession;
            isExportable = other.isExportable;
            payloadSize = other.payloadSize;
            return *this;
        }
    };

    using Uptrc = std::unique_ptr<const CryptoObject>;
    using Uptr = std::unique_ptr<CryptoObject>;

    CryptoObject(const CryptoObjectInfo& crypto_object_info, const CryptoPrimitiveId& primitive_id) {
        crypto_primitive_id_ = primitive_id;
        crypto_object_info_ = crypto_object_info;
    }
    virtual ~CryptoObject () noexcept=default;
    // CryptoObject(bool isSession,bool isExportable):cryobjInfo_(isSession,isExportable){}
    // CryptoObject() = default;
    // template <class ConcreteObject>
    // static ara::core::Result<typename ConcreteObject::Uptrc> Downcast(CryptoObject::Uptrc&& object) noexcept;
    virtual CryptoPrimitiveId::Uptr GetCryptoPrimitiveId () const noexcept {
        return CryptoPrimitiveId::Uptr(new CryptoPrimitiveId(crypto_primitive_id_));
    }

    virtual COIdentifier GetObjectId () const noexcept {
        return crypto_object_info_.objectUid;
    }

    virtual COIdentifier HasDependence () const noexcept {
        return crypto_object_info_.dependencyUid;
    }

    virtual std::size_t GetPayloadSize () const noexcept {
        return crypto_object_info_.payloadSize;
    }

    virtual bool IsExportable () const noexcept {
        return crypto_object_info_.isExportable;
    }

    virtual bool IsSession () const noexcept {
        return crypto_object_info_.isSession;
    }

    virtual netaos::core::Result<void> Save(IOInterface& container) const noexcept {
        return netaos::core::Result<void>::FromError(CryptoErrc::kUnsupported);
    }

    CryptoObject& operator= (const CryptoObject &other)=default;
    CryptoObject& operator= (CryptoObject &&other)=default;
protected:
    CryptoObjectInfo crypto_object_info_;
    CryptoPrimitiveId crypto_primitive_id_;
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_CRYPTO_OBJECT_H_