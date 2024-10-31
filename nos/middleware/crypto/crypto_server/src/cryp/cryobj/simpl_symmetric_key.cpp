#include "cryp/cryobj/simpl_symmetric_key.h"

#include <memory>
#include "common/imp_io_interface.h"
#include "common/crypto_logger.hpp"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

// SimplSymmetricKey(CryptoPrimitiveId::AlgId algId, AllowedUsageFlags allowedUsage, bool isSession, bool isExportable)
// : SymmetricKey(algId, allowedUsage, isSession, isExportable) {

// }

// SimplSymmetricKey::SimplSymmetricKey()
// : key_data_() {

// }

SimplSymmetricKey::~SimplSymmetricKey() {
    CRYP_INFO << "SimplSymmetricKey destructor. add: 0x" << this;
}

SimplSymmetricKey::SimplSymmetricKey(std::vector<uint8_t>& keydata, CryptoObject::CryptoObjectInfo& object_info, CryptoPrimitiveId& primitive_id, AllowedUsageFlags& usage)
: SymmetricKey(object_info, primitive_id, usage)
, key_data_(keydata) {

}

SimplSymmetricKey::SimplSymmetricKey(const SimplSymmetricKey& other)
: SymmetricKey(other.crypto_object_info_, other.crypto_primitive_id_, other.allowed_usage_)
, key_data_(other.key_data_) {

}

SimplSymmetricKey& SimplSymmetricKey::operator= (const SimplSymmetricKey& other) {
    key_data_ = other.key_data_;
    crypto_object_info_ = other.crypto_object_info_;
    crypto_primitive_id_ = other.crypto_primitive_id_;
    allowed_usage_ = other.allowed_usage_;
    return *this;
}

netaos::core::Result<void> SimplSymmetricKey::Save(IOInterface& container) const noexcept {
    std::vector<uint8_t> payload(const_cast<SimplSymmetricKey*>(this)->GetKeyData());
    dynamic_cast<ImpIOInterface&>(container).ioInfo_.usage = const_cast<SimplSymmetricKey*>(this)->GetAllowedUsage();
    dynamic_cast<ImpIOInterface&>(container).ioInfo_.objectSize = const_cast<SimplSymmetricKey*>(this)->GetPayloadSize();
    dynamic_cast<ImpIOInterface&>(container).ioInfo_.isSession = const_cast<SimplSymmetricKey*>(this)->IsSession();
    dynamic_cast<ImpIOInterface&>(container).ioInfo_.objectType = const_cast<SimplSymmetricKey*>(this)->GetObjectId().mCOType;
    dynamic_cast<ImpIOInterface&>(container).ioInfo_.objectUid = const_cast<SimplSymmetricKey*>(this)->GetObjectId().mCouid;
    dynamic_cast<ImpIOInterface&>(container).SetPayload(payload);
    return netaos::core::Result<void>();
}

std::vector<uint8_t> SimplSymmetricKey::GetKeyData() {
    return key_data_;
}

// ara::core::Result<CryptoTransform> ImpSymmetricBlockCipherCtx::GetTransformation() const noexcept {
//     return ara::core::Result<CryptoTransform>::FromValue(transform_);
// };


}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara
