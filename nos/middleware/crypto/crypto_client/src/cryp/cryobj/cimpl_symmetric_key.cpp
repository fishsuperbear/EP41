#include "cimpl_symmetric_key.h"

#include <memory>
#include "client/crypto_cm_client.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

// CimplSymmetricKey(CryptoPrimitiveId::AlgId algId, AllowedUsageFlags allowedUsage, bool isSession, bool isExportable)
// : SymmetricKey(algId, allowedUsage, isSession, isExportable) {

// }

CimplSymmetricKey::CimplSymmetricKey(uint64_t ref, CryptoObject::CryptoObjectInfo& object_info, CryptoPrimitiveId& primitive_id, AllowedUsageFlags& usage)
: SymmetricKey(object_info, primitive_id, usage)
, key_ref_(ref) {

}

CimplSymmetricKey::~CimplSymmetricKey() {
    CryptoCmClient::Instance().ReleaseObject(key_ref_);
}

netaos::core::Result<void> CimplSymmetricKey::Save(IOInterface& container) const noexcept {
    return netaos::core::Result<void>::FromError(CryptoErrc::kUnsupported);
}

CryptoKeyRef CimplSymmetricKey::GetKeyRef() const noexcept  {
    CryptoKeyRef key_ref_info;
    key_ref_info.alg_id = GetCryptoPrimitiveId()->GetPrimitiveId();
    key_ref_info.ref = key_ref_;
    key_ref_info.primitive_id_info.alg_id = crypto_primitive_id_.GetPrimitiveId();
    key_ref_info.crypto_object_info = crypto_object_info_;
    key_ref_info.allowed_usage = allowed_usage_;
    return key_ref_info;
}


// ara::core::Result<CryptoTransform> ImpSymmetricBlockCipherCtx::GetTransformation() const noexcept {
//     return ara::core::Result<CryptoTransform>::FromValue(transform_);
// };


}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara
