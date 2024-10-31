#ifndef IMP_X509_EXTENSIONS_H
#define IMP_X509_EXTENSIONS_H

#include "x509/x509_object.h"
#include "x509/x509_extensions.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
// todo: 这里的extensions是不是指x509的extension信息？
class ImpX509Extensions : public X509Extensions {
public:
    using Uptr = std::unique_ptr<X509Extensions>;


    /// @brief Count number of elements in the sequence.
    /// @return 
    virtual std::size_t Count ();

    virtual X509Provider& MyProvider();

};


}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // X509_EXTENSIONS_H
