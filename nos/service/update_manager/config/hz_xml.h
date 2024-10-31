
#ifndef HOZON_XML_H
#define HOZON_XML_H

#include "tinyxml2.h"

#include <string>
#include <vector>

namespace hozon {
namespace netaos {
namespace update {

using namespace tinyxml2;


class HzXml {
public:
    HzXml(void);
    ~HzXml(void);
    bool ParseXmlFile(const char* xmlFile);

    XMLElement* GetRootElement();
    XMLElement* GetElement(const char* parentTitle, const char* title);
    XMLElement* GetChildElement(XMLElement* root, std::string node_name);
    XMLElement* GetBrotherElement(XMLElement* node, std::string node_name);
    bool GetNodePointerByName(XMLElement* root, std::string &node_name, XMLElement* &node);
    bool GetChildText(XMLElement* root, std::string node_name, std::string &node_text);
    bool GetChildUintHex(XMLElement* root, std::string node_name, uint32_t &node_uint);
    bool GetChildUintDec(XMLElement* root, std::string node_name, uint32_t &node_uint);
    bool GetChildVector(XMLElement* root, std::string node_name, std::vector<uint8_t> &node_vector);
    bool GetElementAttributeValue(XMLElement* Element, const char* AttributeName, std::string& reslut);
    bool GetFirstElementValue(const char* title, std::string& result);
    bool GetNextElementValue(const char* title, std::string& result);
    void Clear();


protected:
    XMLElement* GetFirstElement(const char* ElementMark, XMLElement* pcrElement);

    XMLDocument doc_;
    XMLElement* element_;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // HOZON_XML_H