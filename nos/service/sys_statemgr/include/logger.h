#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "log/include/logging.h"
#include "em/utils/cJSON.h"
#include "sys_statemgr/include/sys_define.h"

namespace hozon {
namespace netaos {
namespace ssm {

class SSMLogger
{
public:
    static SSMLogger& GetInstance() {
        static SSMLogger instance;
        return instance;
    }
    ~SSMLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger>  GetLogger() const { return logger_; }

    bool InitLogger() {
        std::string fpath = SSM_CONFIG_FILE;

        std::string output ="";
        if(fpath.empty()){ return false;}
        std::ifstream in(fpath.c_str(), std::ios::in | std::ios::binary);
        if(in.is_open()){
            while(!in.eof()){ std::string line; getline(in, line, '\n'); output += line;}
            in.close();
        }else{
            std::cout<<"failed to open:"<<fpath<<std::endl;
            return false;
        }
        //
        cJSON *key = nullptr;
        cJSON *root = cJSON_Parse(output.c_str());
        if(!root) { goto ERROR;}
        if (!cJSON_IsObject(root)){ goto ERROR;}

        key = cJSON_GetObjectItem(root, "log_id");
        if (!key) { goto ERROR;}else{
            logid = key->valuestring;
        }
        key = cJSON_GetObjectItem(root, "log_desc");
        if (!key) { goto ERROR;}else{
            logdesc = key->valuestring;
        }
        key = cJSON_GetObjectItem(root, "log_level");
        if (!key) { goto ERROR;}else{
            loglevel = key->valuestring;
        }
        key = cJSON_GetObjectItem(root, "log_mode");
        if (!key) { goto ERROR;}else{
            logmode = key->valuestring;
        }
        key = cJSON_GetObjectItem(root, "log_path");
        if (!key) { goto ERROR;}else{
            logpath = key->valuestring;
        }
        key = cJSON_GetObjectItem(root, "log_max_num");
        if (!key) { goto ERROR;}else{
            logmaxnum = key->valuestring;
        }
        key = cJSON_GetObjectItem(root, "log_max_size");
        if (!key) { goto ERROR;}else{
            logmaxsize = key->valuestring;
        }
        if(root){ cJSON_Delete(root);}

        hozon::netaos::log::InitLogging(logid,logdesc,hozon::netaos::log::LogLevel(std::stoi(loglevel)),
            std::stoi(logmode), logpath, std::stoi(logmaxnum), std::stoi(logmaxsize), true);
        logger_ = hozon::netaos::log::CreateLogger(logid, logdesc,
                        hozon::netaos::log::LogLevel(std::stoi(loglevel)));

        return true;
    ERROR:
        std::cout<<" key or object not found."<<std::endl;
        if(root){ cJSON_Delete(root);}
        return false;
    }
private:
    SSMLogger(){
        if(!InitLogger()){
            hozon::netaos::log::InitLogging("SSM","Sys Statemgr", hozon::netaos::log::LogLevel::kInfo,
                hozon::netaos::log::HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, 20);
            logger_ = hozon::netaos::log::CreateLogger("SSM", "Sys Statemgr",
                        hozon::netaos::log::LogLevel::kInfo);
        }
    };
    std::shared_ptr<hozon::netaos::log::Logger> logger_;

    std::string logid = "SSM";
    std::string logdesc = "Sys Statemgr";
    std::string loglevel = "2";
    std::string logmode = "2";
    std::string logpath = "/opt/usr/log/soc_log/";
    std::string logmaxnum = "10";
    std::string logmaxsize = "10";
};


#define SSM_LOG_HEAD  __FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 <<  "(" << __LINE__ << ") | "

#define SSM_LOG_CRITICAL         (SSMLogger::GetInstance().GetLogger()->LogCritical() << SSM_LOG_HEAD)
#define SSM_LOG_ERROR            (SSMLogger::GetInstance().GetLogger()->LogError() << SSM_LOG_HEAD)
#define SSM_LOG_WARN             (SSMLogger::GetInstance().GetLogger()->LogWarn() << SSM_LOG_HEAD)
#define SSM_LOG_INFO             (SSMLogger::GetInstance().GetLogger()->LogInfo() << SSM_LOG_HEAD)
#define SSM_LOG_DEBUG            (SSMLogger::GetInstance().GetLogger()->LogDebug() << SSM_LOG_HEAD)
#define SSM_LOG_TRACE            (SSMLogger::GetInstance().GetLogger()->LogTrace() << SSM_LOG_HEAD)

template <class T>
static std::string ToString(std::vector<T> t, std::ios_base & (*f)(std::ios_base&), int n = 0)
{
    if (t.size() <= 0) {
        return "";
    }

    std::ostringstream oss;
    int typesize = sizeof(t[0]);
    for (uint i = 0; i < t.size();) {
        if (n) {
            oss << std::setw(n) << std::setfill('0');
        }

        if (1 == typesize) {
            uint8_t item = static_cast<uint8_t>(t[i]);
            oss << f << static_cast<uint16_t>(item);
        }
        else {
            oss << f << t[i];
        }

        ++i;

        if (i < t.size()) {
            oss << " ";
        }
    }

    return oss.str();
}

#define UM_UINT8_VEC_TO_HEX_STRING(vec) ToString<uint8_t>(vec, std::hex, 2)

}
}
}