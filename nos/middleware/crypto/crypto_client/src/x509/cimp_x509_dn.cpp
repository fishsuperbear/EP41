#include "x509/x509_dn.h"

#include "common/crypto_logger.hpp"
#include "x509/cimp_x509_dn.h"
#include "x509/cimp_x509_provider.h"
#include "client/x509_cm_client.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {

bool CimpX509DN::SetDn (std::string dn) {
    return X509CmClient::Instance().SetDn(dn, dn_ref_);
}

std::string CimpX509DN::GetDnString () {
    return  X509CmClient::Instance().GetDnString(dn_ref_);
}

std::string CimpX509DN::GetAttribute(AttributeId id) {
    return X509CmClient::Instance().GetAttribute(id, dn_ref_);
}

std::string CimpX509DN::GetAttribute (AttributeId id, unsigned index){
    return X509CmClient::Instance().GetAttributeWithIndex(id, index,dn_ref_);
}

bool CimpX509DN::SetAttribute(AttributeId id, std::string attribute) {
    return X509CmClient::Instance().SetAttribute(id, attribute,dn_ref_);
}

// CimpX509DN::CimpX509DN() {

// }

CimpX509DN::~CimpX509DN() {

}

X509Provider& CimpX509DN::MyProvider() {
    X509Provider* prov = new CimpX509Provider;
    return *prov;
}

}  // namespace x509
}
}
}

