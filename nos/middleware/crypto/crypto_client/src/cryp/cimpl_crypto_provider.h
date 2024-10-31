#ifndef ARA_CRYPTO_CRYP_IMP_CRYPTO_PROVIDER_H_
#define ARA_CRYPTO_CRYP_IMP_CRYPTO_PROVIDER_H_
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include "core/result.h"
#include "common/base_id_types.h"
#include "cryp/crypto_provider.h"


namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

class CimplCryptoProvider:public CryptoProvider {
public:
    using AlgId = CryptoPrimitiveId::AlgId;
    netaos::core::Result<VolatileTrustedContainer::Uptr> AllocVolatileContainer(std::size_t capacity) noexcept override;
    // virtual ara::core::Result<VolatileTrustedContainer::Uptr> AllocVolatileContainer(std::pair<AlgId, CryptoObjectType> theObjectDef) noexcept = 0;
    AlgId ConvertToAlgId(netaos::core::StringView primitiveName) const noexcept override;
    netaos::core::Result<netaos::core::String> ConvertToAlgName(AlgId algId) const noexcept override;
    // virtual ara::core::Result<AuthCipherCtx::Uptr> CreateAuthCipherCtx(AlgId algId) noexcept = 0;
    netaos::core::Result<DecryptorPrivateCtx::Uptr> CreateDecryptorPrivateCtx(AlgId algId) noexcept override;
    netaos::core::Result<EncryptorPublicCtx::Uptr> CreateEncryptorPublicCtx(AlgId algId) noexcept override;
    // virtual ara::core::Result<Signature::Uptrc> CreateHashDigest(AlgId hashAlgId, ReadOnlyMemRegion value) noexcept = 0;
    netaos::core::Result<HashFunctionCtx::Uptr> CreateHashFunctionCtx(AlgId algId) noexcept override;
    // virtual ara::core::Result<KeyAgreementPrivateCtx::Uptr> CreateKeyAgreementPrivateCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<KeyDecapsulatorPrivateCtx::Uptr> CreateKeyDecapsulatorPrivateCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<KeyDerivationFunctionCtx::Uptr> CreateKeyDerivationFunctionCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<KeyEncapsulatorPublicCtx::Uptr> CreateKeyEncapsulatorPublicCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<MessageAuthnCodeCtx::Uptr> CreateMessageAuthCodeCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<MsgRecoveryPublicCtx::Uptr> CreateMsgRecoveryPublicCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<RandomGeneratorCtx::Uptr> CreateRandomGeneratorCtx(AlgId algId = kAlgIdDefault, bool initialize = true) noexcept = 0;
    // virtual ara::core::Result<SigEncodePrivateCtx::Uptr> CreateSigEncodePrivateCtx(AlgId algId) noexcept = 0;
    // virtual ara::core::Result<Signature::Uptrc> CreateSignature(AlgId signAlgId, ReadOnlyMemRegion value, const RestrictedUseObject& key, AlgId hashAlgId = kAlgIdNone) noexcept = 0;
    netaos::core::Result<SignerPrivateCtx::Uptr> CreateSignerPrivateCtx(AlgId algId) noexcept override;
    // virtual ara::core::Result<StreamCipherCtx::Uptr> CreateStreamCipherCtx(AlgId algId) noexcept = 0;
    netaos::core::Result<SymmetricBlockCipherCtx::Uptr> CreateSymmetricBlockCipherCtx(AlgId algId) noexcept override;
    netaos::core::Result<VerifierPublicCtx::Uptr> CreateVerifierPublicCtx(AlgId algId) noexcept override;
    // virtual ~CryptoProvider() noexcept = default;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > ExportPublicObject(const IOInterface& container, Serializable::FormatId formatId = Serializable::kFormatDefault) noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > ExportSecuredObject(const CryptoObject& object, SymmetricKeyWrapperCtx& transportContext) noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > ExportSecuredObject(const IOInterface& container, SymmetricKeyWrapperCtx& transportContext) noexcept = 0;
    netaos::core::Result<PrivateKey::Uptrc> GeneratePrivateKey(AlgId algId, AllowedUsageFlags allowedUsage, bool isSession, bool isExportable) noexcept override;
    // virtual ara::core::Result<SecretSeed::Uptrc> GenerateSeed(AlgId alg Id, SecretSeed::Usage allowedUsage, bool isSession = true, bool is Exportable = false) noexcept = 0;
    netaos::core::Result<SymmetricKey::Uptrc> GenerateSymmetricKey(AlgId algId,AllowedUsageFlags allowedUsage, bool isSession = true, bool isExportable = false) noexcept override;
    // virtual ara::core::Result<std::size_t> GetPayloadStorageSize(Crypto ObjectType cryptoObjectType, AlgId algId) const noexcept = 0;
    // virtual ara::core::Result<std::size_t> GetSerializedSize(CryptoObject Type cryptoObjectType, AlgId algId, Serializable::FormatId format Id = Serializable::kFormatDefault) const noexcept = 0;
    // virtual ara::core::Result<void> ImportPublicObject(IOInterface& container, ReadOnlyMemRegion serialized, CryptoObjectType expected Object = CryptoObjectType::kUndefined) noexcept = 0;
    // virtual ara::core::Result<void> ImportSecuredObject(IOInterface& container, ReadOnlyMemRegion serialized, SymmetricKeyWrapperCtx& transportContext, bool isExportable = false,
    //                                                     CryptoObjectType expected Object = CryptoObjectType::kUndefined) noexcept = 0;
    // virtual ara::core::Result<CryptoObject::Uptrc> LoadObject(const IOInterface& container) noexcept = 0;
    netaos::core::Result<PrivateKey::Uptrc> LoadPrivateKey(const IOInterface& container) noexcept override;
    // netaos::core::Result<PublicKey::Uptrc> LoadPublicKey(const IOInterface& container) noexcept override;
    // virtual ara::core::Result<SecretSeed::Uptrc> LoadSecretSeed(const IOInterface& container) noexcept = 0;
    // virtual ara::core::Result<SymmetricKey::Uptrc> LoadSymmetricKey(const IOInterface& container) noexcept = 0;

    // static CimplCryptoProvider& Instance();
    
    // CryptoProvider& operator=(const CryptoProvider& other) = default;
    // CryptoProvider& operator=(CryptoProvider&& other) = default;

    bool Init() override;
    bool Deinit() override;

    CimplCryptoProvider();
    ~CimplCryptoProvider();

private:
    // static CimplCryptoProvider *instance_;
    static std::mutex instance_mutex_;
    // CimplCryptoProvider();
    // ~CimplCryptoProvider() noexcept;
    // CimplCryptoProvider(const CimplCryptoProvider&) = delete;
    // CimplCryptoProvider& operator=(const CimplCryptoProvider&) = delete;
    // CimplCryptoProvider(const CimplCryptoProvider&&) = delete;
    // CimplCryptoProvider& operator=(const CimplCryptoProvider&&) = delete;
     const netaos::core::Map<CryptoAlgId, netaos::core::String> toAlgNameMap_ = {
        {kAlgIdCBCAES128, ALG_NAME_CBC_AES128},
        {kAlgIdCBCAES192, ALG_NAME_CBC_AES192},
        {kAlgIdCBCAES256, ALG_NAME_CBC_AES256},
        {kAlgIdGCMAES128, ALG_NAME_GCM_AES128},
        {kAlgIdGCMAES192, ALG_NAME_GCM_AES192},
        {kAlgIdGCMAES256,  ALG_NAME_GCM_AES256},
        {kAlgIdECBAES128,  ALG_NAME_ECB_AES128},
        {kAlgIdMD5,  ALG_NAME_MD5},
        {kAlgIdSHA1,  ALG_NAME_SHA1},
        {kAlgIdSHA256,  ALG_NAME_SHA256},
        {kAlgIdSHA384,  ALG_NAME_SHA384},
        {kAlgIdSHA512,  ALG_NAME_SHA512},
    };
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_IMP_CRYPTO_PROVIDER_H_