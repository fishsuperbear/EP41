#ifndef IMP_X509_PUBLIC_KEY_INFO_H
#define IMP_X509_PUBLIC_KEY_INFO_H

#include "common/serializable.h"
#include "cryp/cryobj/public_key.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
// todo: 依赖密码学原语
class ImpX509PublicKeyInfo : public X509PublicKeyInfo {
public:
    /**
     * @brief Get public key object of the subject.
     * 
     * @param none
     * @return PublicKey
     **/
    virtual hozon::netaos::crypto::cryp::PublicKey::Uptrc GetPublicKey ();

    /**
     * @brief Get an ID of hash algorithm required by current signature algorithm.
     * 
     * @param none
     * @return CryptoAlgId
     **/
    virtual CryptoAlgId GetRequiredHashAlgId () const noexcept=0;

    /**
     * @brief Get the hash size required by current signature algorithm.
     * 
     * @param none
     * @return std::size_t
     **/
    virtual std::size_t GetRequiredHashSize () const noexcept=0;

    /**
     * @brief Get size of the signature value produced and required by the current algorithm.
     * 
     * @param none
     * @return std::size_t
     **/
    virtual std::size_t GetSignatureSize () const noexcept=0;

    /**
     * @brief Get the CryptoPrimitiveId instance of this class.
     * 
     * @param none
     * @return CryptoPrimitiveId
     **/
    virtual hozon::netaos::crypto::cryp::CryptoPrimitiveId::Uptrc GetAlgorithmId()=0;

    /**
     * @brief Verify the sameness of the provided and kept public keys. This method compare the public key values only.
     * 
     * @param publicKey the publicKey to be veryfied
     * @return bool
     **/
    virtual bool IsSameKey (const hozon::netaos::crypto::cryp::PublicKey &publicKey) const noexcept=0;

};


}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // X509_PUBLIC_KEY_INFO_H
