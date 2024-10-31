#include "cimpl_private_key.h"

#include <memory>
#include "client/crypto_cm_client.h"
#include "common/crypto_logger.hpp"
#include "cimpl_public_key.h"
#include "keys/cimpl_io_interface.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

// CimplSymmetricKey(CryptoPrimitiveId::AlgId algId, AllowedUsageFlags allowedUsage, bool isSession, bool isExportable)
// : SymmetricKey(algId, allowedUsage, isSession, isExportable) {

// }

CimplPrivateKey::CimplPrivateKey(uint64_t ref, CryptoObject::CryptoObjectInfo& object_info, CryptoPrimitiveId& primitive_id, AllowedUsageFlags& usage)
: PrivateKey(object_info, primitive_id, usage)
, key_ref_(ref) {

}

CimplPrivateKey::~CimplPrivateKey() {
    CryptoCmClient::Instance().ReleaseObject(key_ref_);
}

netaos::core::Result<PublicKey::Uptrc> CimplPrivateKey::GetPublicKey() const noexcept {
    CryptoKeyRef public_key_ref;
    CryptoObjectInfo object_info;
    cryp::CryptoPrimitiveId primitive_id;
    AllowedUsageFlags usage;
    int32_t ipc_res = CryptoCmClient::Instance().GetPublicKey(GetKeyRef(), public_key_ref);

    auto uptr = std::make_unique<CimplPublicKey>(public_key_ref.ref, object_info, primitive_id, usage);

    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        return netaos::core::Result<PublicKey::Uptrc>::FromValue(std::move(uptr));
    }
    CRYP_INFO<<"GetPublicKey public_key_ref.ref:"<<public_key_ref.ref;
    return netaos::core::Result<PublicKey::Uptrc>::FromValue(std::move(uptr));
}

bool CimplPrivateKey::CheckKey(bool strongCheck) const noexcept {
    return true;
}

netaos::core::Result<void> CimplPrivateKey::Save(IOInterface& container) const noexcept {
    int32_t ipc_res = CryptoCmClient::Instance().Save(key_ref_, dynamic_cast<CimplOInterface*>(&container)->getContainer().ref);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        CRYP_ERROR << "CimplPrivateKey Save error, res : " << ipc_res;
        return netaos::core::Result<void>::FromError(CryptoErrc::kUnsupported);
    }
    return netaos::core::Result<void>();
}

CryptoKeyRef CimplPrivateKey::GetKeyRef() const noexcept  {
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
