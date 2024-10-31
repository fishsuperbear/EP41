#ifndef X509_EXTENSIONS_H
#define X509_EXTENSIONS_H

#include "x509_object.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
// todo: 这里的extensions是不是指x509的extension信息？
class X509Extensions : public X509Object {
public:
    using Uptr = std::unique_ptr<X509Extensions>;


    /// @brief Count number of elements in the sequence.
    /// @return 
    virtual std::size_t Count () = 0;

};


}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // X509_EXTENSIONS_H
