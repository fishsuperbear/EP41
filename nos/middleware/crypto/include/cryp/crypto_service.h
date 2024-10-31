#ifndef ARA_CRYPTO_CRYP_CRYPTO_SERVICE_H_
#define ARA_CRYPTO_CRYP_CRYPTO_SERVICE_H_

#include "extension_service.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class CryptoService : public ExtensionService{
public:
    using Uptr = std::unique_ptr<CryptoService>;

    struct CryptoServiceInfo {
        uint64_t block_size;
        uint64_t max_input_size;
        uint64_t max_output_size;

        CryptoServiceInfo()
        : block_size(0)
        , max_input_size(0)
        , max_output_size(0) {

        }

        CryptoServiceInfo(uint64_t _block_size, uint64_t _max_input_size, uint64_t _max_output_size)
        : block_size(_block_size)
        , max_input_size(_max_input_size)
        , max_output_size(_max_output_size) {

        }

        CryptoServiceInfo(const CryptoServiceInfo& other)
        : block_size(other.block_size)
        , max_input_size(other.max_input_size)
        , max_output_size(other.max_output_size) {

        }

        CryptoServiceInfo& operator= (const CryptoServiceInfo& other) {
            block_size = other.block_size;
            max_input_size = other.max_input_size;
            max_output_size = other.max_output_size;
            return *this;
        }
    };

    virtual std::size_t GetBlockSize() const noexcept = 0;
    virtual std::size_t GetMaxInputSize(bool suppressPadding = false) const noexcept = 0;
    virtual std::size_t GetMaxOutputSize(bool suppressPadding = false) const noexcept = 0;
private:

};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_CRYPTO_SERVICE_H_