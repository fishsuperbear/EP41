#include "update_manager/config/hz_xml.h"


namespace hozon {
namespace netaos {
namespace update {



HzXml::HzXml(void)
{
}

HzXml::~HzXml(void)
{
}

bool HzXml::ParseXmlFile(const char* xmlFile)
{
    return doc_.LoadFile(xmlFile) == XML_SUCCESS ? true : false;
}

XMLElement*
HzXml::GetElement(const char* parentTitle, const char* title)
{
    XMLNode* node = doc_.FirstChildElement(parentTitle);
    for (node = node->FirstChild(); node; node = node->NextSibling()) {
        if (!strcmp(title, node->Value())) {
            return node->ToElement();
        }
    }

    return 0;
}

bool
HzXml::GetElementAttributeValue(XMLElement* Element, const char* AttributeName, std::string& reslut)
{
    if (Element->Attribute(AttributeName)) {
        reslut = Element->Attribute(AttributeName);
        return true;
    }

    return false;
}

bool
HzXml::GetFirstElementValue(const char* title, std::string& result)
{
    if (!title)
        return 0;
    XMLElement* element(0);
    element = doc_.RootElement();
    element = GetFirstElement(title, element);
    if (element) {
        element_ = element;
        result = element_->GetText();
        return true;
    }

    return false;
}

bool
HzXml::GetNextElementValue(const char* title, std::string& result)
{
    result = "";
    element_= element_->NextSiblingElement(title);
    if (element_) {
        result = element_->GetText();
        return true;
    }
    return false;
}

XMLElement*
HzXml::GetRootElement()
{
    return doc_.RootElement();
}

XMLElement*
HzXml::GetChildElement(XMLElement* root, std::string node_name)
{
    if (nullptr == root) {
        return nullptr;
    }

    XMLElement *node = nullptr;
    if (GetNodePointerByName(root, node_name, node)) {
        if (nullptr != node) {
            return node;
        }
    }

    return nullptr;
}

XMLElement*
HzXml::GetBrotherElement(XMLElement* node, std::string node_name)
{
    if (NULL == node) {
        return NULL;
    }

    return node->NextSiblingElement();
}

bool
HzXml::GetNodePointerByName(XMLElement* root, std::string &node_name, XMLElement* &node)
{
    XMLElement* element = nullptr;
    for (element = root->FirstChildElement(); element != nullptr; element = element->NextSiblingElement()) {
        // if(GetNodePointerByName(element, node_name, node)) return true;
        if (node_name == element->Value()) {
            node = element;
            return true;
        }
    }

    return false;
}

bool
HzXml::GetChildText(XMLElement* root, std::string node_name, std::string &node_text)
{
    if (nullptr == root) {
        node_text = "";
        return false;
    }

    XMLElement *node = nullptr;
    if (GetNodePointerByName(root, node_name, node)) {
        if (nullptr != node) {
            // 如果存在key，不存在value，会进入这个case，，例如<ProcessProportion></ProcessProportion>
            // 会导致异常崩溃
            if (node->GetText() == nullptr)
            {
                node_text = "";
                return false;
            }
            node_text = node->GetText();
            return true;
        }
    }
    node_text = "";
    return false;
}

bool
HzXml::GetChildUintHex(XMLElement* root, std::string node_name, uint32_t &node_uint)
{
    bool ret = false;
    if (nullptr == root) {
        return ret;
    }

    XMLElement *node = nullptr;
    std::string strValue;
    if (!GetNodePointerByName(root, node_name, node)) {
        return ret;
    }

    if (nullptr == node) {
        return ret;
    }

    strValue = node->GetText();
    uint32_t  tmp = 0;
    for (long unsigned int i = 0; i < (strValue.size()/3 + 1); i++) {
        tmp = tmp << 8 | (std::stoi(&strValue[i*3], 0, 16));
    }
    node_uint = tmp;
    ret = true;
    return ret;
}

bool
HzXml::GetChildUintDec(XMLElement* root, std::string node_name, uint32_t &node_uint)
{
    bool ret = false;
    if (nullptr == root) {
        return ret;
    }

    XMLElement *node = nullptr;
    std::string strValue;
    if (!GetNodePointerByName(root, node_name, node)) {
        return ret;
    }

    if (nullptr == node) {
        return ret;
    }

    strValue = node->GetText();
    uint32_t  tmp = 0;
    for (long unsigned int i = 0; i < (strValue.size()/3 + 1); i++) {
        tmp = tmp << 8 | (uint32_t)(std::stoi(&strValue[i*3], 0, 10));
    }
    node_uint = tmp;
    ret = true;
    return ret;
}

bool
HzXml::GetChildVector(XMLElement* root, std::string node_name, std::vector<uint8_t> &node_vector)
{
    bool ret = false;
    if (nullptr == root) {
        return ret;
    }

    XMLElement *node = nullptr;
    std::string strValue;
    if (!GetNodePointerByName(root, node_name, node)) {
        return ret;
    }

    if (nullptr == node) {
        return ret;
    }

    strValue = node->GetText();
    for (long unsigned int i = 0; i < strValue.size()/3+1; i++) {
        node_vector.push_back((uint8_t)std::stoi(&strValue[i*3], 0, 16));
    }
    ret = true;
    return ret;
}

void
HzXml::Clear()
{
    doc_.Clear();
}

XMLElement*
HzXml::GetFirstElement(const char* ElementMark, XMLElement* pcrElement)
{
    XMLElement* element = pcrElement;
    while (element) {
        if (strcmp(element->Value(), ElementMark) == 0) {
            //printf("%s\r\n",element_tmp->Value());
            return element;
        }
        else {
            XMLElement* nextElement = element->FirstChildElement();
            while (nextElement) {
                //printf("%s\r\n",nextElement->Value());
                if (strcmp(nextElement->Value(), ElementMark) == 0) {
                    return nextElement;
                }
                else {
                    XMLElement* reElement = NULL;
                    reElement = GetFirstElement(ElementMark, nextElement);
                    if (reElement) {
                        return reElement;
                    }
                }
                nextElement = nextElement->NextSiblingElement();
            }
        }
        element = element->NextSiblingElement();
    }
    return NULL;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon