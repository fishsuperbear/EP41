#ifndef ARA_CRYPTO_CRYP_PUBLIC_KEY_H_
#define ARA_CRYPTO_CRYP_PUBLIC_KEY_H_

#include <string>
#include "core/result.h"
#include "common/base_id_types.h"
#include "cryp/cryobj/restricted_use_object.h"
#include "cryp/hash_function_ctx.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class PublicKey:public RestrictedUseObject{
public:

    using Uptrc = std::unique_ptr<const PublicKey>;
    PublicKey(const CryptoObjectInfo& object_info, const CryptoPrimitiveId& primitive_id, const AllowedUsageFlags& usage)
    : RestrictedUseObject(usage, object_info, primitive_id) {

    }
    
    // virtual ~PublicKey() = default;
    virtual bool CheckKey(bool strongCheck = true) const noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > HashPublicKey(HashFunctionCtx& hashFunc) const noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t> > HashPublicKey(HashFunctionCtx& hashFunc) const noexcept = 0;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> HashPublicKey(HashFunctionCtx& hashFunc) const noexcept;
    static const CryptoObjectType kObjectType = CryptoObjectType::kPublicKey;
    // virtual std::string get_myName() const{
    //     std::string name("PublicKey");
    //     return name;
    // } ;

private:

};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_PUBLIC_KEY_H_