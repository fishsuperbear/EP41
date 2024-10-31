#include "common/crypto_error_domain.h"

namespace hozon {
namespace netaos {
namespace crypto {

namespace internal {
hozon::netaos::crypto::CryptoErrorDomain CRYPTO_ERROR_DOMAIN;
}

hozon::netaos::crypto::CryptoErrorDomain const& GetCryptoErrorDomain() noexcept {
    return internal::CRYPTO_ERROR_DOMAIN;
}

hozon::netaos::core::ErrorCode MakeErrorCode(CryptoErrorDomain::Errc code, netaos::core::ErrorDomain::SupportDataType data) noexcept {
    return hozon::netaos::core::ErrorCode(static_cast<hozon::netaos::core::ErrorDomain::CodeType>(code), GetCryptoErrorDomain(), data); 
}

}
}
}