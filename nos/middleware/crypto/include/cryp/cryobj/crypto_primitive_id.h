#ifndef ARA_CRYPTO_CRYP_CRYPTO_PRIMITEVED_ID_H_
#define ARA_CRYPTO_CRYP_CRYPTO_PRIMITEVED_ID_H_
#include "core/string_view.h"
#include "common/base_id_types.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

class CryptoPrimitiveId {
public:
    using AlgId = hozon::netaos::crypto::CryptoAlgId;
    using Uptrc = std::unique_ptr<const CryptoPrimitiveId>;
    using Uptr = std::unique_ptr<CryptoPrimitiveId>;

    struct PrimitiveIdInfo {
        AlgId alg_id;

        PrimitiveIdInfo()
        : alg_id(kAlgIdUndefined) {

        }

        PrimitiveIdInfo(const AlgId& alg_id)
        : alg_id(alg_id) {

        }

        PrimitiveIdInfo(const PrimitiveIdInfo& other)
        : alg_id(other.alg_id) {

        }

        PrimitiveIdInfo& operator = (const PrimitiveIdInfo& other) {
            alg_id = other.alg_id;
            return *this;
        }
    };

    CryptoPrimitiveId()
    : primitive_id_info_(kAlgIdUndefined) {
    }

    CryptoPrimitiveId(const AlgId& alg_id)
    : primitive_id_info_(alg_id) {

    }

    CryptoPrimitiveId(const PrimitiveIdInfo& primitive_id_info)
    : primitive_id_info_(primitive_id_info) {

    }

    CryptoPrimitiveId(const CryptoPrimitiveId& other)
    : primitive_id_info_(other.primitive_id_info_) {

    }

    CryptoPrimitiveId& operator= (const CryptoPrimitiveId& other) {
        primitive_id_info_ = other.primitive_id_info_;
        return *this;
    }

    virtual ~CryptoPrimitiveId () noexcept=default;
    virtual AlgId GetPrimitiveId() const noexcept {
        return primitive_id_info_.alg_id;
    }

    virtual const std::string GetPrimitiveName() const noexcept {
        // TODO
        return "UNKNOW";
    }

    // CryptoPrimitiveId& operator= (const CryptoPrimitiveId &other)=default;
    // CryptoPrimitiveId& operator= (CryptoPrimitiveId &&other)=default;

protected:
   PrimitiveIdInfo primitive_id_info_;
 
};

}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_CRYPTO_PRIMITEVED_ID_H_