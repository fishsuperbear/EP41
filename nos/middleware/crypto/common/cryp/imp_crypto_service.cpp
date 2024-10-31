#include "cryp/imp_crypto_service.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

std::size_t ImpCryptoService::GetBlockSize() const noexcept {
    return blockSize_;
};

std::size_t ImpCryptoService::GetMaxInputSize(bool suppressPadding) const noexcept{
    return maxInputSize_;
};

std::size_t ImpCryptoService::GetMaxOutputSize(bool suppressPadding) const noexcept{
    return maOutputSize_;
};

std::size_t ImpCryptoService::GetActualKeyBitLength () const noexcept{
    return actualKeyBitLength_;
};

CryptoObjectUid ImpCryptoService::GetActualKeyCOUID () const noexcept{
    return cryObjUid_;
};

AllowedUsageFlags ImpCryptoService::GetAllowedUsage () const noexcept {
    return allowedUsageFlags_;
};

std::size_t ImpCryptoService::GetMaxKeyBitLength () const noexcept {
    return maxKeyBitLength_;
};

std::size_t ImpCryptoService::GetMinKeyBitLength () const noexcept {
    return minKeyBitLength_;
};

bool ImpCryptoService::IsKeyBitLengthSupported(std::size_t keyBitLength) const noexcept {
    return ((keyBitLength >= GetMinKeyBitLength()) && (keyBitLength <= GetMaxKeyBitLength()));
};

bool ImpCryptoService::IsKeyAvailable () const noexcept {
    return true; //TODO
};

}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara
