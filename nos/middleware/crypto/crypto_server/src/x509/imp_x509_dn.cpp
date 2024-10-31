#include "x509/x509_dn.h"

#include "common/crypto_logger.hpp"
#include "x509/imp_x509_dn.h"
#include "x509/imp_x509_provider.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {

std::map<X509DN::AttributeId, std::string> AttributeId_NID_MAP = {
    {X509DN::AttributeId::kCountry, "C"},
    {X509DN::AttributeId::kStreet, "ST"},
    {X509DN::AttributeId::kLocality, "L"},
    {X509DN::AttributeId::kOrganization, "O"},
    {X509DN::AttributeId::kOrgUnit, "OU"},
    {X509DN::AttributeId::kCommonName, "CN"},
    {X509DN::AttributeId::kEmail, "emailAddress"}
};

std::map<std::string, X509DN::AttributeId> NID_AttributeId_MAP = {
    {"C", X509DN::AttributeId::kCountry},
    {"ST", X509DN::AttributeId::kStreet},
    {"L", X509DN::AttributeId::kLocality},
    {"O", X509DN::AttributeId::kOrganization},
    {"OU", X509DN::AttributeId::kOrgUnit},
    {"CN", X509DN::AttributeId::kCommonName},
    {"emailAddress", X509DN::AttributeId::kEmail}
};

bool ImpX509DN::SetDn (std::string dn) {
    CRYP_INFO << "ImpX509DN SetDn";
    // 先清空
    attributeMap.clear();

    std::vector<std::string> nidList;
    Stringsplit(dn, '/', nidList);
    for (auto it: nidList) {
        size_t pos = it.find('=');
        size_t size = it.size();
        std::string commonName = it.substr(0, pos);
        std::string value = it.substr(pos + 1, size);
        auto tmpIt = NID_AttributeId_MAP.find(commonName);
        if (tmpIt != NID_AttributeId_MAP.end()) {
            attributeMap.insert(std::make_pair(tmpIt->second, value));
            CRYP_INFO << " Id:" << static_cast<int>(tmpIt->second) << " " << "value:" << value;
        } else {
            CRYP_ERROR << "set unsuport dn nid error: " << commonName << ":" << value;
        }
    }
    return true;
}

std::string ImpX509DN::GetDnString () {
    std::string resStr = "";
    if (attributeMap.empty()) {
        return resStr;
    }

    if (!attributeMap[X509DN::AttributeId::kCountry].empty()) {
        resStr.append("/").append(AttributeId_NID_MAP[X509DN::AttributeId::kCountry]).append("=").append(attributeMap[X509DN::AttributeId::kCountry]);
    }
    if (!attributeMap[X509DN::AttributeId::kStreet].empty()) {
        resStr.append("/").append(AttributeId_NID_MAP[X509DN::AttributeId::kStreet]).append("=").append(attributeMap[X509DN::AttributeId::kStreet]);
    }
    if (!attributeMap[X509DN::AttributeId::kLocality].empty()) {
        resStr.append("/").append(AttributeId_NID_MAP[X509DN::AttributeId::kLocality]).append("=").append(attributeMap[X509DN::AttributeId::kLocality]);
    }
    if (!attributeMap[X509DN::AttributeId::kOrganization].empty()) {
        resStr.append("/").append(AttributeId_NID_MAP[X509DN::AttributeId::kOrganization]).append("=").append(attributeMap[X509DN::AttributeId::kOrganization]);
    }
    if (!attributeMap[X509DN::AttributeId::kOrgUnit].empty()) {
        resStr.append("/").append(AttributeId_NID_MAP[X509DN::AttributeId::kOrgUnit]).append("=").append(attributeMap[X509DN::AttributeId::kOrgUnit]);
    }
    if (!attributeMap[X509DN::AttributeId::kCommonName].empty()) {
        resStr.append("/").append(AttributeId_NID_MAP[X509DN::AttributeId::kCommonName]).append("=").append(attributeMap[X509DN::AttributeId::kCommonName]);
    }if (!attributeMap[X509DN::AttributeId::kEmail].empty()) {
        resStr.append("/").append(AttributeId_NID_MAP[X509DN::AttributeId::kEmail]).append("=").append(attributeMap[X509DN::AttributeId::kEmail]);
    }
    // for (auto it: attributeMap) {
    //     auto itTmp = AttributeId_NID_MAP.find(it.first);
    //     if (itTmp != AttributeId_NID_MAP.end())
    //     {
    //         resStr.append("/").append(itTmp->second).append("=").append(it.second);
    //     }else {
    //         CRYP_ERROR << "get unsuport dn nid error: " << (std::uint32_t)it.first << ":" << it.second;
    //     }
    // }
    return resStr;
}

std::string ImpX509DN::GetAttribute(AttributeId id) {
    if (attributeMap.find(id) != attributeMap.end()) {
        return attributeMap[id];
    }
    return "";
}

bool ImpX509DN::SetAttribute(AttributeId id, std::string attribute) {
    attributeMap[id] = attribute;
    return true; 
}

void ImpX509DN::Stringsplit(std::string &str, char splist, std::vector<std::string> &res) {
    if (str == "") return;
    std::string strs = str.substr(1, str.size()) + splist;
    size_t pos = strs.find(splist);

    while (pos != strs.npos)
    {
        std::string tmp = strs.substr(0, pos);
        res.push_back(tmp);
        strs = strs.substr(pos + 1, str.size());
        pos = strs.find(splist);
    }
}

X509Provider& ImpX509DN::MyProvider() {
    X509Provider* prov = new ImpX509Provider;
    return *prov;
}

}  // namespace x509
}
}
}

