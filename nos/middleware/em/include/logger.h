#pragma once

#include <iostream>
#include <fstream>
#include "log/include/logging.h"
#include "em/utils/cJSON.h"
#include "em/include/define.h"

namespace hozon {
namespace netaos {
namespace em {

class EManagerLogger
{
public:
    static EManagerLogger& GetInstance() {
        static EManagerLogger instance;
        return instance;
    }
    ~EManagerLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger>  GetLogger() const { return logger_; }

    bool InitLogger() {
        std::string fpath = EXECMAGER_CONFIG_FILE;
#ifdef EM_DEBUG_ON
        char* tmp_app_dir = getenv(DEV_ENVRION_PROC_DIR);
        char* tmp_conf_dir = getenv(DEV_ENVRION_CONF_DIR);
        if(tmp_app_dir && tmp_conf_dir){
            fpath = std::string(tmp_conf_dir) +"/"+ DEV_EXECMAGER_CONFIG_FILE;
        }
#endif
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
            std::stoi(logmode), logpath, std::stoi(logmaxnum), std::stoi(logmaxsize));
        logger_ = hozon::netaos::log::CreateLogger(logid,logdesc,hozon::netaos::log::LogLevel(std::stoi(loglevel)));

        return true;
    ERROR:
        std::cout<<" key or object not found."<<std::endl;
        if(root){ cJSON_Delete(root);}
        return false;
    }
private:
    EManagerLogger(){
        if(!InitLogger()){
            hozon::netaos::log::InitLogging("EM","NETAOS EM",hozon::netaos::log::LogLevel::kInfo,
                hozon::netaos::log::HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, 20);
            logger_ = hozon::netaos::log::CreateLogger("EM", "NETAOS EM",
                        hozon::netaos::log::LogLevel::kInfo);
        }
    };
    std::shared_ptr<hozon::netaos::log::Logger> logger_;

    std::string logid;
    std::string logdesc;
    std::string loglevel;
    std::string logmode;
    std::string logpath;
    std::string logmaxnum;
    std::string logmaxsize;
};

#define EM_LOG_HEAD ""
#define EM_LOG_CRITICAL_WITH_HEAD       LOG_CRITICAL << EM_LOG_HEAD
#define EM_LOG_ERROR_WITH_HEAD          LOG_ERROR << EM_LOG_HEAD
#define EM_LOG_WARN_WITH_HEAD           LOG_WARN << EM_LOG_HEAD
#define EM_LOG_INFO_WITH_HEAD           LOG_INFO << EM_LOG_HEAD
#define EM_LOG_DEBUG_WITH_HEAD          LOG_DEBUG << EM_LOG_HEAD
#define EM_LOG_TRACE_WITH_HEAD          LOG_TRACE << EM_LOG_HEAD

#define LOG_CRITICAL         (EManagerLogger::GetInstance().GetLogger()->LogCritical())
#define LOG_ERROR            (EManagerLogger::GetInstance().GetLogger()->LogError())
#define LOG_WARN             (EManagerLogger::GetInstance().GetLogger()->LogWarn())
#define LOG_INFO             (EManagerLogger::GetInstance().GetLogger()->LogInfo())
#define LOG_DEBUG            (EManagerLogger::GetInstance().GetLogger()->LogDebug())
#define LOG_TRACE            (EManagerLogger::GetInstance().GetLogger()->LogTrace())
#define BR                   (EManagerLogger::GetInstance().GetLogger()->LogTrace())

#define CONFIG_LOG_HEAD getpid() << " " << (long int)syscall(186) << " " << strrchr(__FILE__, '/') + 1 << "@" << __FUNCTION__ << "(" << __LINE__ << ") | "

#define EM_LOG_CRITICAL_WITH_HEAD_FUNC LOG_CRITICAL << CONFIG_LOG_HEAD
#define EM_LOG_ERROR_WITH_HEAD_FUNC LOG_ERROR << CONFIG_LOG_HEAD
#define EM_LOG_WARN_WITH_HEAD_FUNC LOG_WARN << CONFIG_LOG_HEAD
#define EM_LOG_INFO_WITH_HEAD_FUNC LOG_INFO << CONFIG_LOG_HEAD
#define EM_LOG_DEBUG_WITH_HEAD_FUNC LOG_DEBUG << CONFIG_LOG_HEAD
#define EM_LOG_TRACE_WITH_HEAD_FUNC LOG_TRACE << CONFIG_LOG_HEAD

}
}
}