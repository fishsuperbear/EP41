#ifndef IMP_X509_DN_H
#define IMP_X509_DN_H

#include <iostream>
#include <map>
#include <vector>

#include "x509/x509_object.h"
#include "x509/x509_dn.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
class ImpX509DN : public X509DN {
public:

    /// @brief Set whole Distinguished Name (DN) from a single string
    /// @param dn 
    bool SetDn (std::string dn);

    /// @brief  Get the whole Distinguished Name (DN) as a single string
    /// @return 
    std::string GetDnString ();

    /// @brief Get DN attribute by its ID
    /// @param id 
    /// @return 
    std::string GetAttribute (AttributeId id);

    /// @brief Return DN attribute by its ID and sequential index
    /// @param id 
    /// @param index 
    /// @return 
    std::string GetAttribute (AttributeId id, unsigned index);

    /// @brief Set DN attribute by its ID
    /// @param id 
    /// @param attribute 
    /// @return 
    virtual bool SetAttribute (AttributeId id, std::string attribute);

    /// @brief Set DN attribute by its ID and sequential index
    /// @param id 
    /// @param index 
    /// @param attribute AttributeId_NID_MAP
    /// @return 
    // virtual bool SetAttribute (AttributeId id, unsigned index, std::string attribute);

    virtual X509Provider& MyProvider();

    ImpX509DN() {};
    ~ImpX509DN() {};

private:
    void Stringsplit(std::string &str, char splist, std::vector<std::string> &res);

};


}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // X509_DN_H
