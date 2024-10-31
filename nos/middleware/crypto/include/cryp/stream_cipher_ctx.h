#ifndef ARA_CRYPTO_CRYP_STREAM_CIPHER_CTX_H_
#define ARA_CRYPTO_CRYP_STREAM_CIPHER_CTX_H_
#include "core/result.h"
#include "common/mem_region.h"
#include "cryp/cryobj/secret_seed.h"
#include "cryp/crypto_context.h"
#include "cryp/block_service.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class StreamCipherCtx:public CryptoContext{
public:
    using Uptr = std::unique_ptr<StreamCipherCtx>;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> Sign(ReadOnlyMemRegion value, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept;
    virtual std::size_t CountBytesInCache() const noexcept = 0;
    std::size_t EstimateMaxInputSize(std::size_t outputCapacity) const noexcept;
    std::size_t EstimateRequiredCapacity(std::size_t inputSize, bool isFinal = false) const noexcept;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte>> FinishBytes(ReadOnlyMemRegion in) noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t>> FinishBytes(ReadOnlyMemRegion in) noexcept = 0;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> FinishBytes(ReadOnlyMemRegion in) noexcept;
    virtual BlockService::Uptr GetBlockService() const noexcept = 0;
    virtual bool IsBytewiseMode () const noexcept=0;
    virtual netaos::core::Result<CryptoTransform> GetTransformation() const noexcept = 0;
    virtual bool IsSeekableMode() const noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte>> ProcessBlocks(ReadOnlyMemRegion in) noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t>> ProcessBlocks(ReadOnlyMemRegion in) noexcept = 0;
    virtual netaos::core::Result<void> ProcessBlocks(ReadWriteMemRegion inOut) noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte>> ProcessBytes(ReadOnlyMemRegion in) noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t>> ProcessBytes(ReadOnlyMemRegion in) noexcept = 0;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> ProcessBytes(ReadOnlyMemRegion in) noexcept;
    virtual netaos::core::Result<void> Reset() noexcept = 0;
    virtual netaos::core::Result<void> Seek(std::int64_t offset, bool fromBegin = true) noexcept = 0;
    virtual netaos::core::Result<void> SetKey(const SymmetricKey& key, CryptoTransform transform = CryptoTransform::kEncrypt) noexcept = 0;
    virtual netaos::core::Result<void> Start(ReadOnlyMemRegion iv = ReadOnlyMemRegion()) noexcept = 0;
    virtual netaos::core::Result<void> Start(const SecretSeed& iv) noexcept = 0;

   private:
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_STREAM_CIPHER_CTX_H_