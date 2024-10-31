#ifndef ARA_CRYPTO_CRYP_IMP_PUBLIC_KEY_H_
#define ARA_CRYPTO_CRYP_IMP_PUBLIC_KEY_H_

#include <cstddef>
#include <string>
// #include "openssl/core_names.h"
#include "openssl/rsa.h"
#include "core/result.h"
#include "common/base_id_types.h"
#include "cryp/cryobj/public_key.h"
#include "cryp/cryobj/restricted_use_object.h"
#include "cryp/hash_function_ctx.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class ImpPublicKey:public PublicKey{
public:

    using Uptrc = std::unique_ptr<const ImpPublicKey>;
    static const CryptoObjectType kObjectType = CryptoObjectType::kPublicKey;
    ImpPublicKey(EVP_PKEY *pkey, CryptoObjectInfo& object_info, CryptoPrimitiveId& primitive_id, AllowedUsageFlags& usage)
    : PublicKey(object_info, primitive_id, usage)
    , pkey_(pkey) {

    }

    ~ImpPublicKey();
    // ara::core::Result<ImpPublicKey::Uptrc> GetPublicKey() const noexcept override;

    bool CheckKey(bool strongCheck = true) const noexcept override;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > HashPublicKey(HashFunctionCtx& hashFunc) const noexcept = 0;

    netaos::core::Result<netaos::core::Vector<uint8_t> > HashPublicKey(HashFunctionCtx& hashFunc) const noexcept override;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> HashPublicKey(HashFunctionCtx& hashFunc) const noexcept;

    netaos::core::Result<void> Save(IOInterface& container) const noexcept override;

    EVP_PKEY * get_pkey(){
        // CRYP_INFO<<"get_pkey called.";
        // CRYP_INFO<<"pkey_ addr:";
        std::cout<< "get_pkey called."<<std::endl;
        return pkey_;
    };

    // virtual std::string get_myName() const{
    //     std::string name("ImpPublicKey");
    //     return name;
    // };

private:
    EVP_PKEY *pkey_ = NULL; // private-public pair

};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara


#endif  // #define ARA_CRYPTO_CRYP_IMP_PUBLIC_KEY_H_