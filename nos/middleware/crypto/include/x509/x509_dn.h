#ifndef X509_DN_H
#define X509_DN_H

#include <iostream>
#include <map>
#include <vector>
#include "x509_object.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
class X509DN : public X509Object {
public:
    using Uptr = std::unique_ptr<X509DN>;
    using Uptrc = std::unique_ptr<const X509DN>;

    enum class AttributeId : std::uint32_t {
        kCommonName= 0, // Common Name.
        kCountry= 1, //Country.
        kState= 2, //State.
        kLocality= 3, //Locality.
        kOrganization= 4, // Organization.
        kOrgUnit= 5, // Organization Unit.
        kStreet= 6, // Street.
        kPostalCode= 7, // Postal Code.
        kTitle= 8, // Title.
        kSurname= 9, // Surname.
        kGivenName= 10, // Given Name.
        kInitials= 11, // Initials.
        kPseudonym= 12, // Pseudonym.
        kGenerationQualifier= 13, // Generation Qualifier.
        kDomainComponent= 14, // Domain Component.
        kDnQualifier= 15, // Distinguished Name Qualifier.
        kEmail= 16, // E-mail.
        kUri= 17, // URI.
        kDns= 18, // DNS.
        kHostName= 19, // Host Name (UNSTRUCTUREDNAME)
        kIpAddress= 20, // IP Address (UNSTRUCTUREDADDRESS)
        kSerialNumbers= 21, // Serial Numbers.
        kUserId= 22 // User ID.
    };

    std::map<AttributeId, std::string> attributeMap;

    /// @brief Set whole Distinguished Name (DN) from a single string
    /// @param dn 
    virtual bool SetDn (std::string dn)  = 0;

    /// @brief  Get the whole Distinguished Name (DN) as a single string
    /// @return 
    virtual std::string GetDnString () = 0;

    /// @brief Get DN attribute by its ID
    /// @param id 
    /// @return 
    virtual std::string GetAttribute (AttributeId id) = 0;

    /// @brief Return DN attribute by its ID and sequential index
    /// @param id 
    /// @param index 
    /// @return 
    // virtual std::string GetAttribute (AttributeId id, unsigned index) = 0;

    /// @brief Set DN attribute by its ID
    /// @param id 
    /// @param attribute 
    /// @return 
    virtual bool SetAttribute (AttributeId id, std::string attribute) = 0;

    /// @brief Set DN attribute by its ID and sequential index
    /// @param id 
    /// @param index 
    /// @param attribute AttributeId_NID_MAP
    /// @return 
    // virtual bool SetAttribute (AttributeId id, unsigned index, std::string attribute);

    X509DN(){}
    virtual ~X509DN(){};
private:
    // void Stringsplit(std::string &str, char splist, std::vector<std::string> &res);

};


}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // X509_DN_H
