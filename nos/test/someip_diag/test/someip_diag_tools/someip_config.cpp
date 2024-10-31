
#include <cstring>
#include <iomanip>
#include <iostream>
#include "json/json.h"

#include "someip_config.h"

SomeIPConfig* SomeIPConfig::instancePtr_ = nullptr;
std::mutex SomeIPConfig::instance_mtx_;

SomeIPConfig* SomeIPConfig::Instance() {
    if (nullptr == instancePtr_) {
        std::lock_guard<std::mutex> lck(instance_mtx_);
        if (nullptr == instancePtr_) {
            instancePtr_ = new SomeIPConfig();
        }
    }
    return instancePtr_;
}

SomeIPConfig::SomeIPConfig()
{
    udsReqs_.clear();
}

SomeIPConfig::~SomeIPConfig() {}

bool SomeIPConfig::LoadConfig() {
    std::string config_path = "someip_tools.json";
    std::cout << "SomeIPConfig::LoadConfig configPath: " << config_path.c_str() << std::endl;
    char* json = GetJsonAll(config_path.c_str());
    if (NULL == json) {
        return false;
    }
    return ParseJSON(json);
}

char* SomeIPConfig::GetJsonAll(const char* fname) {
    FILE* fp;
    char* str;
    char txt[5000];
    int filesize;
    if ((fp = fopen(fname, "r")) == NULL) {
        std::cout << "<SomeIPConfig> open file " << fname << " fail!" << std::endl;
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    filesize = ftell(fp);

    str = (char*)malloc(filesize + 1);  //malloc more size, or strcat will coredump
    if (!str) {
        std::cout << "malloc error." << std::endl;
        return NULL;
    }
    memset(str, 0, filesize + 1);

    rewind(fp);
    while ((fgets(txt, 1000, fp)) != NULL) {
        strcat(str, txt);
    }
    fclose(fp);

    return str;
}

bool SomeIPConfig::ParseJSON(char* jsonstr) {
    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
    Json::Value rootValue;
    JSONCPP_STRING errs;

    bool res = reader->parse(jsonstr, jsonstr + strlen(jsonstr), &rootValue, &errs);

    if (res && errs.empty()) {
        
        Json::Value & sourceAddressValue = rootValue["DiagSourceAddress"];
        source_adr_ = static_cast<uint16_t>(std::strtoul(sourceAddressValue.asString().c_str(), 0, 0));

        Json::Value & targetAddressValue = rootValue["DiagTragetAddress"];
        target_adr_ = static_cast<uint16_t>(std::strtoul(targetAddressValue.asString().c_str(), 0, 0));

        Json::Value diagCommands = rootValue["DiagCommandList"];
        uint16_t id = 1;
        for (auto command : diagCommands) {
            std::vector<uint8_t> req;
            std::string commandString = command.asString();
            std::stringstream ss(commandString);
            std::string byteString;
            while (ss >> byteString) {
                req.push_back(std::stoi(byteString, nullptr, 16));
            }
            udsReqs_[id] = req;
            ++id;
        }

    } else {
        std::cout << "json parse error!" << std::endl;
        return false;
    }

    return true;
}

uint16_t SomeIPConfig::getSourceAdr()
{
    return source_adr_;
}

uint16_t SomeIPConfig::getTargetAdr()
{
    return target_adr_;
}

// 打印udsReqs_
void SomeIPConfig::printUdsReqs() const {
    for (const auto& it : udsReqs_) {
        std::cout << "ID: " << it.first << ", Request: ";
        for (const auto& byte : it.second) {
            std::cout << std::hex << static_cast<int>(byte) << " ";
        }
        std::cout << std::endl;
    }
}

// 返回序号对应的请求
std::vector<uint8_t> SomeIPConfig::getUdsReqById(uint16_t id) const {
    auto it = udsReqs_.find(id);
    if (it != udsReqs_.end()) {
        return it->second;
    }

    return {};
}

// 设置序号对应的响应结果
void SomeIPConfig::setRespResultById(uint16_t id, bool result) {
    auto it = udsReqs_.find(id);
    if (it != udsReqs_.end()) {
        if (result)
        {
            it->second = {0x00, 0x00, 0x00};
        }
        else
        {
            it->second = {0x0F, 0x0F, 0x0F};
        }
    }
}
