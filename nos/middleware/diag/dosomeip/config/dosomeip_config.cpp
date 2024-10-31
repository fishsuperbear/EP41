
#include <cstring>
#include <iomanip>
#include <iostream>
#include "json/json.h"

#include "config/dosomeip_config.h"
#include "diag/dosomeip/log/dosomeip_logger.h"

namespace hozon {
namespace netaos {
namespace diag {

DoSomeIPConfig* DoSomeIPConfig::instancePtr_ = nullptr;
std::mutex DoSomeIPConfig::instance_mtx_;

DoSomeIPConfig* DoSomeIPConfig::Instance() {
    if (nullptr == instancePtr_) {
        std::lock_guard<std::mutex> lck(instance_mtx_);
        if (nullptr == instancePtr_) {
            instancePtr_ = new DoSomeIPConfig();
        }
    }
    return instancePtr_;
}

DoSomeIPConfig::DoSomeIPConfig() : initTimeout_{10}, respTimeout_{5} {}

DoSomeIPConfig::~DoSomeIPConfig() {}

bool DoSomeIPConfig::LoadConfig() {

#ifdef BUILD_FOR_MDC
    std::string config_path = "/opt/usr/diag_update/mdc-llvm/conf/dosomeip.json";
#elif BUILD_FOR_J5
    std::string config_path = "/userdata/diag_update/j5/conf/dosomeip.json";
#elif BUILD_FOR_ORIN
    std::string config_path = "/app/runtime_service/diag_server/conf/dosomeip.json";
#else
    std::string config_path = "/app/runtime_service/diag_server/conf/dosomeip.json";
#endif

    DS_INFO << "DoSomeIPConfig::LoadConfig configPath: " << config_path.c_str();
    char* json = GetJsonAll(config_path.c_str());
    if (NULL == json) {
        return false;
    }
    return ParseJSON(json);
}

char* DoSomeIPConfig::GetJsonAll(const char* fname) {
    FILE* fp;
    char* str;
    char txt[5000];
    int filesize;
    if ((fp = fopen(fname, "r")) == NULL) {
        DS_ERROR << "<DoSomeIPConfig> open file " << fname << " fail!";
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    filesize = ftell(fp);

    str = (char*)malloc(filesize + 1);  //malloc more size, or strcat will coredump
    if (!str) {
        DS_ERROR << "malloc error.";
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

bool DoSomeIPConfig::ParseJSON(char* jsonstr) {
    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
    Json::Value rootValue;
    JSONCPP_STRING errs;

    bool res = reader->parse(jsonstr, jsonstr + strlen(jsonstr), &rootValue, &errs);

    if (res && errs.empty()) {
        initTimeout_ = rootValue["InitMaxTimeout"].asUInt();
        respTimeout_ = rootValue["ResponseMaxTimeout"].asUInt();
        DS_DEBUG << "<DoSomeIPConfig> ParseJSON: InitMaxTimeout is " << initTimeout_;
        DS_DEBUG << "<DoSomeIPConfig> ParseJSON: ResponseMaxTimeout is " << respTimeout_;

        doipAddressList_.clear();
        Json::Value & doipAddressValue = rootValue["DoipAddressList"];
        for (uint32_t i = 0; i < doipAddressValue.size(); i++) {
            doipAddressList_.emplace_back(static_cast<uint16_t>(std::strtoul(doipAddressValue[i].asString().c_str(), 0, 0)));
        }

        docanAddressList_.clear();
        Json::Value & docanAddressValue = rootValue["DocanAddressList"];
        for (uint32_t i = 0; i < docanAddressValue.size(); i++) {
            docanAddressList_.emplace_back(static_cast<uint16_t>(std::strtoul(docanAddressValue[i].asString().c_str(), 0, 0)));
        }

        Json::Value & doSomeIPProxyAddressValue = rootValue["DoSomeIPProxyAddress"];
        doSomeIPProxyAddress_ = static_cast<uint16_t>(std::strtoul(doSomeIPProxyAddressValue.asString().c_str(), 0, 0));

    } else {
        DS_ERROR << "json parse error!";
        return false;
    }

    return true;
}

uint16_t DoSomeIPConfig::GetInitMaxTimeout() {
    return initTimeout_;
}
uint16_t DoSomeIPConfig::GetResponseMaxTimeout() {
    return respTimeout_;
}

std::vector<uint16_t> DoSomeIPConfig::GetDoipAddressList()
{
    return doipAddressList_;
}
std::vector<uint16_t> DoSomeIPConfig::GetDocanAddressList()
{
    return docanAddressList_;
}

uint16_t DoSomeIPConfig::GetDoSomeIPProxyAddress()
{
    return doSomeIPProxyAddress_;
}

bool DoSomeIPConfig::IsDoIPAddress(const uint16_t& address)
{
    auto itr = std::find(doipAddressList_.begin(), doipAddressList_.end(), address);
    if (itr != doipAddressList_.end()) {
        return true;
    }

    return false;
}

bool DoSomeIPConfig::IsDoCanAddress(const uint16_t& address)
{
    auto itr = std::find(docanAddressList_.begin(), docanAddressList_.end(), address);
    if (itr != docanAddressList_.end()) {
        return true;
    }
    return false;
}

bool DoSomeIPConfig::IsDoSomeipProxyAddress(const uint16_t& address)
{
    return doSomeIPProxyAddress_ == address ? true : false;
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */
