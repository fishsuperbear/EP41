#pragma once

#include <vector>
#include "common/inner_types.h"
#include "cryp/cryobj/crypto_primitive_id.h"
#include "cryp/cryobj/symmetric_key.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

class SimplSymmetricKey: public SymmetricKey {
public:
    // using Uptrc = std::unique_ptr<const SymmetricKey>;
    // static const CryptoObjectType kObjectType = CryptoObjectType::kSymmetricKey;
    // SimplSymmetricKey(CryptoPrimitiveId::AlgId algId, AllowedUsageFlags allowedUsage, bool isSession, bool isExportable)
    // : SymmetricKey(algId, allowedUsage, isSession, isExportable) {

    // }

    SimplSymmetricKey();
    SimplSymmetricKey(std::vector<uint8_t>& keydata, CryptoObjectInfo& object_info, cryp::CryptoPrimitiveId& primitive_id, AllowedUsageFlags& usage);
    SimplSymmetricKey(const SimplSymmetricKey& other);
    SimplSymmetricKey& operator= (const SimplSymmetricKey& other);

    ~SimplSymmetricKey();

    netaos::core::Result<void> Save(IOInterface& container) const noexcept override;
    std::vector<uint8_t> GetKeyData();

private:

    std::vector<uint8_t> key_data_;

//  ImpCryptoPrimitiveId impPrimitiveId_;
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara