#ifndef X509_OBJECT_H
#define X509_OBJECT_H

#include "common/serializable.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {

class X509Provider;
class X509Object : public Serializable {
public:
    //  @brief Get a reference to X.509 Provider of this object.
    // 
    //  @param None
    //  @return X509Provider&
    virtual X509Provider& MyProvider() = 0;

    virtual ~X509Object() {};


};


}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // X509_OBJECT_H
