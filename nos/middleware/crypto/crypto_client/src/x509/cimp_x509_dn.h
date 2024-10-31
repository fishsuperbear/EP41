#ifndef CIMP_X509_DN_H
#define CIMP_X509_DN_H

#include <iostream>
#include <map>
#include <vector>

#include "x509/x509_object.h"
#include "x509/x509_dn.h"
#include "common/inner_types.h"


namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
class CimpX509DN : public X509DN {
public:
    CimpX509DN(X509DNRef dn_ref):X509DN(),dn_ref_(dn_ref){}
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
    bool SetAttribute (AttributeId id, std::string attribute);

    /// @brief Set DN attribute by its ID and sequential index
    /// @param id 
    /// @param index 
    /// @param attribute AttributeId_NID_MAP
    /// @return 
    // virtual bool SetAttribute (AttributeId id, unsigned index, std::string attribute);

    X509DNRef getX509DnRef() {
        return dn_ref_;
    }

    X509Provider& MyProvider();

    CimpX509DN(){};
    ~CimpX509DN() override;

private:
    // void Stringsplit(std::string &str, char splist, std::vector<std::string> &res);
    X509DNRef dn_ref_;

};


}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // X509_DN_H
