#ifndef BASIC_CERT_INFO_H
#define BASIC_CERT_INFO_H

#include <stdint.h>

#include <memory>

#include "x509_dn.h"
#include "x509_object.h"
#include "x509_public_key_info.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
class BasicCertInfo : public X509Object {
 public:
  using KeyConstraints = std::uint32_t;
  static const KeyConstraints kConstrCrlSign = 0x0200;
  static const KeyConstraints kConstrDataEncipherment = 0x1000;
  static const KeyConstraints kConstrDecipherOnly = 0x0080;
  static const KeyConstraints kConstrDigitalSignature = 0x8000;
  static const KeyConstraints kConstrEncipherOnly = 0x0100;
  static const KeyConstraints kConstrKeyAgreement = 0x0800;
  static const KeyConstraints kConstrKeyCertSign = 0x0400;
  static const KeyConstraints kConstrKeyEncipherment = 0x2000;
  static const KeyConstraints kConstrNonRepudiation = 0x4000;
  static const KeyConstraints kConstrNone = 0;

  std::string subjectName;
  X509DN::Uptr pSubjectDN;

  //  @brief Get the key constraints for the key associated with this PKCS#10
  //  object. 返回的是复合取值，使用时需要自行匹配
  //  @param None
  //  @return KeyConstraints
  // virtual KeyConstraints GetConstraints() = 0;

  //  @brief Get the constraint on the path length defined in the Basic
  //  Constraints extension. This extension is used to limit the length of a
  //  cert chain that may be issued from that CA.
  //  @param None
  //  @return KeyConstraints
  virtual uint32_t GetPathLimit() = 0;

  //  @brief Check whether the CA attribute of X509v3 Basic Constraints is true
  //  (i.e. pathlen=0).
  //
  //  @param None
  //  @return bool
  virtual bool IsCa() = 0;

  //  @brief Get the subject DN.
  //
  //  @param None
  //  @return X509DN&
  virtual const X509DN& SubjectDn() = 0;

  //  @brief Load the subject public key information object to realm of
  //  specified crypto provider.
  //
  //  @param cryp::CryptoProvider::Uptr
  //  @return X509PublicKeyInfo&
  // virtual const X509PublicKeyInfo& SubjectPubKey (cryp::CryptoProvider::Uptr
  // cryptoProvider=nullptr) const noexcept=0;
  // virtual const X509PublicKeyInfo& SubjectPubKey(
  //     std::unique_ptr<void> cryptoProvider);

  // BasicCertInfo();
  virtual ~BasicCertInfo() {};
};

}  // namespace x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // BASIC_CERT_INFO_H
