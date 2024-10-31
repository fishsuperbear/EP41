#ifndef ARA_CRYPTO_CRYP_CRYPTO_PROVIDER_H_
#define ARA_CRYPTO_CRYP_CRYPTO_PROVIDER_H_
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include "core/result.h"

#include "common/base_id_types.h"
#include "common/volatile_trusted_container.h"
#include "common/io_interface.h"
#include "common/mem_region.h"
#include "common/serializable.h"

// #include "auth_cipher_ctx.h"
#include "cryp/cryobj/symmetric_key.h"
#include "cryp/cryobj/private_key.h"
#include "cryp/hash_function_ctx.h"
#include "cryp/verifier_public_ctx.h"
#include "cryp/signer_private_ctx.h"
#include "cryp/stream_cipher_ctx.h"
#include "cryp/stream_cipher_ctx.h"
#include "cryp/random_generator_ctx.h"
#include "cryp/message_authn_code_ctx.h"
#include "cryp/encryptor_public_ctx.h"
#include "cryp/decryptor_private_ctx.h"
#include "cryp/cryobj/crypto_primitive_id.h"
#include "cryp/cryobj/crypto_object.h"
#include "cryp/key_derivation_function_ctx.h"
#include "cryp/key_decapsulator_private_ctx.h"
#include "cryp/key_encapsulator_public_ctx.h"
#include "cryp/key_agreement_private_ctx.h"
#include "cryp/symmetric_key_wrapper_ctx.h"
#include "cryp/msg_recovery_public_ctx.h"
#include "cryp/sig_encode_private_ctx.h"
#include "cryp/symmetric_block_cipher_ctx.h"


namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

// using namespace netaos::crypto;
using namespace hozon::netaos::crypto;

class CryptoProvider {
public:
    using AlgId = CryptoPrimitiveId::AlgId;
    using Uptr = std::unique_ptr<CryptoProvider>;
    virtual netaos::core::Result<VolatileTrustedContainer::Uptr> AllocVolatileContainer(std::size_t capacity = 0) noexcept = 0;
    // virtual ara::core::Result<VolatileTrustedContainer::Uptr> Alloc VolatileContainer(std::pair<AlgId, CryptoObjectType> theObjectDef) noexcept = 0;
    virtual AlgId ConvertToAlgId(netaos::core::StringView primitiveName) const noexcept = 0;
    virtual netaos::core::Result<netaos::core::String> ConvertToAlgName(AlgId algId) const noexcept = 0;
    // virtual ara::core::Result<AuthCipherCtx::Uptr> CreateAuthCipherCtx(AlgId algId) noexcept = 0;
    virtual netaos::core::Result<DecryptorPrivateCtx::Uptr> CreateDecryptorPrivateCtx(AlgId algId) noexcept = 0;
    virtual netaos::core::Result<EncryptorPublicCtx::Uptr> CreateEncryptorPublicCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<Signature::Uptrc> CreateHashDigest(AlgId hashAlgId, ReadOnlyMemRegion value) noexcept = 0;
    virtual netaos::core::Result<HashFunctionCtx::Uptr> CreateHashFunctionCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<KeyAgreementPrivateCtx::Uptr> CreateKeyAgreementPrivateCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<KeyDecapsulatorPrivateCtx::Uptr> CreateKeyDecapsulatorPrivateCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<KeyDerivationFunctionCtx::Uptr> CreateKeyDerivationFunctionCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<KeyEncapsulatorPublicCtx::Uptr> CreateKeyEncapsulatorPublicCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<MessageAuthnCodeCtx::Uptr> CreateMessageAuthCodeCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<MsgRecoveryPublicCtx::Uptr> CreateMsgRecoveryPublicCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<RandomGeneratorCtx::Uptr> CreateRandomGeneratorCtx(AlgId algId = kAlgIdDefault, bool initialize = true) noexcept = 0;
    // virtual ara::core::Result<SigEncodePrivateCtx::Uptr> CreateSigEncodePrivateCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<Signature::Uptrc> CreateSignature(AlgId signAlgId, ReadOnlyMemRegion value, const RestrictedUseObject& key, AlgId hashAlgId = kAlgIdNone) noexcept = 0;
    virtual netaos::core::Result<SignerPrivateCtx::Uptr> CreateSignerPrivateCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<StreamCipherCtx::Uptr> CreateStreamCipherCtx(AlgId algId) noexcept = 0;
    virtual netaos::core::Result<SymmetricBlockCipherCtx::Uptr> CreateSymmetricBlockCipherCtx(AlgId algId) noexcept = 0;
    virtual netaos::core::Result<VerifierPublicCtx::Uptr> CreateVerifierPublicCtx(AlgId algId) noexcept = 0;
    // virtual ~CryptoProvider() noexcept = default;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > ExportPublicObject(const IOInterface& container, Serializable::FormatId formatId = Serializable::kFormatDefault) noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > ExportSecuredObject(const CryptoObject& object, SymmetricKeyWrapperCtx& transportContext) noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > ExportSecuredObject(const IOInterface& container, SymmetricKeyWrapperCtx& transportContext) noexcept = 0;
    virtual netaos::core::Result<PrivateKey::Uptrc> GeneratePrivateKey(AlgId algId, AllowedUsageFlags allowedUsage, bool isSession = false, bool isExportable = false) noexcept = 0;
    // virtual ara::core::Result<SecretSeed::Uptrc> GenerateSeed(AlgId algId, SecretSeed::Usage allowedUsage, bool isSession = true, bool is Exportable = false) noexcept = 0;
    virtual netaos::core::Result<SymmetricKey::Uptrc> GenerateSymmetricKey(AlgId algId, AllowedUsageFlags allowedUsage, bool isSession = true, bool isExportable = false) noexcept = 0;
    // virtual ara::core::Result<std::size_t> GetPayloadStorageSize(Crypto ObjectType cryptoObjectType, AlgId algId) const noexcept = 0;
    // virtual ara::core::Result<std::size_t> GetSerializedSize(CryptoObject Type cryptoObjectType, AlgId algId, Serializable::FormatId format Id = Serializable::kFormatDefault) const noexcept = 0;
    // virtual ara::core::Result<void> ImportPublicObject(IOInterface& container, ReadOnlyMemRegion serialized, CryptoObjectType expected Object = CryptoObjectType::kUndefined) noexcept = 0;
    // virtual ara::core::Result<void> ImportSecuredObject(IOInterface& container, ReadOnlyMemRegion serialized, SymmetricKeyWrapperCtx& transportContext, bool isExportable = false,
    //                                                     CryptoObjectType expected Object = CryptoObjectType::kUndefined) noexcept = 0;
    // virtual ara::core::Result<CryptoObject::Uptrc> LoadObject(const IOInterface& container) noexcept = 0;
    virtual netaos::core::Result<PrivateKey::Uptrc> LoadPrivateKey(const IOInterface& container) noexcept = 0;
    // virtual netaos::core::Result<PublicKey::Uptrc> LoadPublicKey(const IOInterface& container) noexcept = 0;
    // virtual ara::core::Result<SecretSeed::Uptrc> LoadSecretSeed(const IOInterface& container) noexcept = 0;
    // virtual ara::core::Result<SymmetricKey::Uptrc> LoadSymmetricKey(const IOInterface& container) noexcept = 0;
    
    CryptoProvider& operator=(const CryptoProvider& other) = default;
    CryptoProvider& operator=(CryptoProvider&& other) = default;
    virtual ~CryptoProvider(){};
    virtual bool Init() = 0;
    virtual bool Deinit() = 0;
   private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_CRYPTO_PROVIDER_H_