#pragma once

#include "common/inner_types.h"
#include "cryp/cryobj/crypto_primitive_id.h"
#include "cryp/cryobj/public_key.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

class CimplPublicKey: public PublicKey {
public:
    // using Uptrc = std::unique_ptr<const SymmetricKey>;
    // static const CryptoObjectType kObjectType = CryptoObjectType::kSymmetricKey;
    // CimplSymmetricKey(CryptoPrimitiveId::AlgId algId, AllowedUsageFlags allowedUsage, bool isSession, bool isExportable)
    // : SymmetricKey(algId, allowedUsage, isSession, isExportable) {

    // }

    CimplPublicKey(uint64_t ref, CryptoObjectInfo& object_info, cryp::CryptoPrimitiveId& primitive_id, AllowedUsageFlags& usage);

    ~CimplPublicKey();

    netaos::core::Result<netaos::core::Vector<uint8_t> > HashPublicKey(HashFunctionCtx& hashFunc) const noexcept override;

    bool CheckKey(bool strongCheck = true) const noexcept override;

    netaos::core::Result<void> Save(IOInterface& container) const noexcept override;
    CryptoKeyRef GetKeyRef() const noexcept;

private:

    uint64_t key_ref_;

//  ImpCryptoPrimitiveId impPrimitiveId_;
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara