#ifndef ARA_CRYPTO_COMMON_SERIALIZABLE_H_
#define ARA_CRYPTO_COMMON_SERIALIZABLE_H_

#include <vector>
#include <cstddef>
#include "base_id_types.h"

namespace hozon {
namespace netaos {
namespace crypto {

class Serializable {
public:
using FormatId = std::uint32_t;
    // virtual ~Serializable() noexcept = default;
    // // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > ExportPublicly(FormatId formatId = kFormatDefault) const noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<uint8_t> > ExportPublicly(FormatId formatId = kFormatDefault) const noexcept = 0;
    // template <typename Alloc = uint8_t>
    // ByteVector<Alloc> ExportPublicly(FormatId formatId = kFormatDefault) const noexcept;
    // // template <typename Alloc = <implementation - defined>>
    // // ara::core::Result<ByteVector<Alloc>> ExportPublicly(FormatId format Id = kFormatDefault) const noexcept;
    // Serializable& operator= (const Serializable &other)=default;
    // Serializable& operator= (Serializable &&other)=default;
    
    static const FormatId kFormatDefault = 0;
    static const FormatId kFormatRawValueOnly = 1;
    static const FormatId kFormatDerEncoded = 2;
    static const FormatId kFormatPemEncoded = 3;
private:
};

}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_COMMON_SERIALIZABLE_H_