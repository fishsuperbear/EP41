#ifndef ARA_CRYPTO_CRYP_BLOCK_SERVICE_H_
#define ARA_CRYPTO_CRYP_BLOCK_SERVICE_H_

#include "extension_service.h"
namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class BlockService : public ExtensionService{
public:
    using Uptr = std::unique_ptr<BlockService>;

    struct BlockServiceInfo {
        uint64_t block_size;
        uint64_t iv_size;
        bool is_valid_iv_size;

        BlockServiceInfo()
        : block_size(0)
        , iv_size(0)
        , is_valid_iv_size(false) {
            
        }

        BlockServiceInfo(uint64_t _block_size, uint64_t _iv_size, bool _is_valid_iv_size)
        : block_size(_block_size)
        , iv_size(_iv_size)
        , is_valid_iv_size(_is_valid_iv_size) {
            
        }

        BlockServiceInfo(const BlockServiceInfo& other)
        : block_size(other.block_size)
        , iv_size(other.iv_size)
        , is_valid_iv_size(other.is_valid_iv_size) {
            
        }

        BlockServiceInfo& operator= (const BlockServiceInfo& other) {
            block_size = other.block_size;
            iv_size = other.iv_size;
            is_valid_iv_size = other.is_valid_iv_size;
            return *this;
        }
    };

    BlockService() noexcept;
    virtual ~BlockService() = default;
    virtual std::size_t GetBlockSize () const noexcept=0;
    virtual std::size_t GetIvSize () const noexcept=0;
    virtual bool IsValidIvSize (std::size_t ivSize) const noexcept=0;


private:
    BlockService(const BlockService&) = delete;
    BlockService& operator=(const BlockService&) = delete;
    BlockService(const BlockService&&) = delete;
    BlockService& operator=(const BlockService&&) = delete;
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_BLOCK_SERVICE_H_