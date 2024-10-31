/************************************************************************************
*  Copyright (c) Hozon Technologies Co., Ltd. 2022-2030. All rights reserved.       *
*                                                                                   *
*  @file     ara_logging.cpp                                                        *
*  @brief    Implement of logging functions                                         *
*  Details.                                                                         *
*                                                                                   *
*  @version  0.0.0.1                                                                *
*                                                                                   *
*-----------------------------------------------------------------------------------*
*  Change History :                                                                 *
*  <Date>     | <Version> | <Author>       | <Description>                          *
*-----------------------------------------------------------------------------------*
*  2022/06/15 | 0.0.0.1   | YangPeng      | Create file                             *
*-----------------------------------------------------------------------------------*
*                                                                                   *
*************************************************************************************/

#include "logging.h"
#include "logstream.h"
#include "log_manager.hpp"
#include "hz_operation_logger.hpp"
#include "json/json.h"

#include <vector>
#include <fstream>
#include <thread>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>


namespace hozon {
namespace netaos {
namespace log {

static bool bStopFlag = false;
static bool bInitFlag = false;

static uint32_t iLoggerCount = 0;
std::thread* pChangeLevelThread = nullptr;

void setLogLevel(
    const std::string appId, // the log id of application, input "IGNORE" when don't want set special application log level
    const std::string ctxId, // the log id of application, input "IGNORE" when don't want set special application log level
    const LogLevel level
) noexcept
{
    if (appId == "IGNORE") {
        if (ctxId == "IGNORE") {
            HzLogManager::GetInstance()->setAllCtxLogLevel(level);
        }
        else {
            HzLogManager::GetInstance()->setCtxLogLevel(ctxId, level);
        }
    }
    else {
        if (ctxId == "IGNORE") {
            HzLogManager::GetInstance()->setAppLogLevel(appId, level);
        }
        else {
            HzLogManager::GetInstance()->setSpecifiedLogLevel(appId, ctxId, level);
        }
    }
}

void Quit()
{
    // std::cout << "Logger Quit iLoggerCount: " << iLoggerCount << std::endl;
    if ((--iLoggerCount) == 0) {
        bStopFlag = true;
        int fd = socket(AF_INET, SOCK_DGRAM, 0);
        struct sockaddr_in remote_addr;
        memset(&remote_addr, 0, sizeof(remote_addr));
        remote_addr.sin_family = AF_INET;
        remote_addr.sin_port = htons(58297);
        remote_addr.sin_addr.s_addr = inet_addr("224.0.0.55");
        std::string msg = "";
        // std::cout << "Logger Quit sendto event for thread join. " << std::endl;
        errno = 0;
        if (sendto(fd,  msg.c_str(), msg.size(), 0, (struct sockaddr *)&remote_addr, sizeof(remote_addr))) {
            std::cout << "Logger Quit sendto event failed!, errno: " << errno << "\n";
        }
        if (pChangeLevelThread != nullptr && pChangeLevelThread->joinable()) {
            pChangeLevelThread->join();
            delete pChangeLevelThread;
        }
        close(fd);
    }
}

void setLogLevelByCmd(std::string rcvCMD)
{
    auto pos1 = rcvCMD.find(".");
    auto pos2 = rcvCMD.find(":");

    if ((pos1 != rcvCMD.npos) && (pos2 != rcvCMD.npos)) {
        std::string appId = rcvCMD.substr(0, pos1);
        std::string ctxId = rcvCMD.substr(pos1 + 1, pos2 -pos1 - 1);
        std::string strLogLevel = rcvCMD.substr(pos2 + 1, rcvCMD.size() - pos2 -1);

        // std::cout << "++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        // std::cout << "appId: " << appId << ", ctxId: " << ctxId << ", strLogLevel: " << strLogLevel << std::endl;
        // std::cout << "++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

        LogLevel level = LogLevel::kCritical;

        if (strLogLevel.find("kTrace") != strLogLevel.npos) {
            level = LogLevel::kTrace;
        }
        else if (strLogLevel.find("kDebug") != strLogLevel.npos) {
            level = LogLevel::kDebug;
        }
        else if (strLogLevel.find("kInfo") != strLogLevel.npos) {
            level = LogLevel::kInfo;
        }
        else if (strLogLevel.find("kWarn") != strLogLevel.npos) {
            level = LogLevel::kWarn;
        }
        else if (strLogLevel.find("kError") != strLogLevel.npos) {
            level = LogLevel::kError;
        }
        else if (strLogLevel.find("kCritical") != strLogLevel.npos) {
            level = LogLevel::kCritical;
        }
        else if (strLogLevel.find("kOff") != strLogLevel.npos) {
            level = LogLevel::kOff;
        }
        else {
            std::cout << "SetLogLevelByCmd failed! rcvCMD level format exeption! strLogLevel: "
                      << strLogLevel.c_str() << std::endl;
            return;
        }

        setLogLevel(appId, ctxId, level);
    }
}

void updateEnvLoggerSettings()
{
    /* "HZ_SET_LOG_LEVEL" Env check...*/
    char* env_str = NULL;

    env_str = getenv("HZ_SET_LOG_LEVEL");

    if (NULL != env_str) {
        //std::cout << "env_str: " << env_str << std::endl;
        setLogLevelByCmd(env_str);
    }
}


void LogSetBySocketThread()
{
    /*Following check the "hz_set_log_level" CMD from socket...*/
    pthread_setname_np(pthread_self(), "LogSetBySocketThread");
    struct ip_mreq mreq;
    struct sockaddr_in local_addr;
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        std::cout << "socket failed\n";
        return;
    }

    int on = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) < 0) {
        std::cout << "fail to set socket SO_REUSEADDR\n";
        close(fd);
        return;
    }

    // multicast receive dynamic control log level cmd
    mreq.imr_multiaddr.s_addr = inet_addr("224.0.0.55");
    mreq.imr_interface.s_addr = inet_addr("0.0.0.0");
    errno = 0;
    if (setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        std::cout << "fail to join multicast group, errno " << errno << "\n";
        close(fd);
        return;
    }

    memset(&local_addr, 0, sizeof(struct sockaddr_in));
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = htons(58297);
    local_addr.sin_addr.s_addr = inet_addr("0.0.0.0");

    errno = 0;
    if (bind(fd, (struct sockaddr *)&local_addr, sizeof(local_addr)) < 0) {
        std::cout << "bind failed, errno " << errno << "\n";
        close(fd);
        return;
    }

    struct timeval tv;
    tv.tv_sec = 2;
    tv.tv_usec = 0;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        std::cout << "socket option  SO_RCVTIMEO not support,errno  " << errno << "\n";
        close(fd);
        return;
    }

    char buf[128] = {'\0'};
    while(!bStopFlag) {
        auto str_len = recvfrom(fd, buf, sizeof(buf), 0, NULL, 0);
        if (str_len < 0) {
            continue;
        }
        buf[str_len] = '\0';
        setLogLevelByCmd(buf);
    }

    // std::cout << "LogSetBySocketThread closed!\n";
    close(fd);
}


void InitLogging(
    std::string appId,
    std::string appDescription,
    LogLevel appLogLevel,
    std::uint32_t outputMode,
    std::string directoryPath,
    std::uint32_t maxLogFileNum,
    std::uint64_t maxSizeOfLogFile,
    bool isMain,
    bool pureLogFormat
    ) noexcept
{
    if (bInitFlag)
    {
        return;
    }

    // Call log manager init to init the application relate info
    HzLogManager::GetInstance()->InitLogging(appId, appDescription, appLogLevel,
                                            outputMode,directoryPath, maxLogFileNum,
                                            (maxSizeOfLogFile * 1024 * 1024), pureLogFormat);
    updateEnvLoggerSettings();

    if (nullptr == pChangeLevelThread) {
        pChangeLevelThread = new std::thread(LogSetBySocketThread);
    }

    if (isMain)
    {
        bInitFlag = true;
    }
}

void InitLogging(std::string logCfgFile) noexcept
{
    std::ifstream cfgFile(logCfgFile.c_str(), std::ios::in);
    if (!cfgFile.good()) {
        std::cout << "logCfgFile read error! please check the config file!" << std::endl;
        return;
    }

    std::string appId{"DEFAULT_APP"}; // Log application identification
    std::string appDescription{"default application"};    // Log application description
    LogLevel appLogLevel{LogLevel::kError};  // Log level of application
    std::uint32_t outputMode{0};    // Log mode
    std::string directoryPath{"/opt/usr/log/soc_log/"}; // Log stored directory path
    std::uint32_t  maxLogFileStoredNum{10};
    std::uint64_t   maxSizeOfEachLogFile{20 * 1024 * 1024};
    bool isMain{false};
    bool pureLogFormat{false};

    std::string strJson((std::istreambuf_iterator<char>(cfgFile)),
                         std::istreambuf_iterator<char>());

    // std::cout << "log config content: " << strJson << std::endl;

    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
    Json::Value  rootValue;
    JSONCPP_STRING errs;

    bool res = reader->parse(strJson.c_str(), strJson.c_str() + strlen(strJson.c_str()), &rootValue, &errs);
    if (res && errs.empty()) {
        appId = rootValue["appId"].asString();
        appDescription = rootValue["appDescription"].asString();
        directoryPath = rootValue["logFile"].asString();

        std::string strAppLogLevel = rootValue["logLevel"].asString();
        std::string strOutputMode = rootValue["logMode"].asString();
        maxLogFileStoredNum = (std::uint32_t)rootValue["maxLogFileStoredNum"].asUInt();
        maxSizeOfEachLogFile = (std::uint64_t)(rootValue["maxSizeOfEachLogFile"].asUInt() * (1024 * 1024));
        if(!rootValue["isMain"].isNull())
        {
            isMain = rootValue["isMain"].asBool();
        }
        pureLogFormat = rootValue["pureLogFormat"].asBool();

        outputMode = 0;
        if (strOutputMode.find("console") != strOutputMode.npos) {
            outputMode += HZ_LOG2CONSOLE;
        }

        if (strOutputMode.find("file") != strOutputMode.npos) {
            outputMode += HZ_LOG2FILE;
        }

        if (strOutputMode.find("logservice") != strOutputMode.npos) {
            outputMode += HZ_LOG2LOGSERVICE;
        }

        if (0 == outputMode) {
            outputMode += HZ_LOG2FILE;
        }

        if (strAppLogLevel == "kTrace") {
            appLogLevel = LogLevel::kTrace;
        }
        else if (strAppLogLevel == "kDebug") {
            appLogLevel = LogLevel::kDebug;
        }
        else if (strAppLogLevel == "kInfo") {
            appLogLevel = LogLevel::kInfo;
        }
        else if (strAppLogLevel == "kWarn") {
            appLogLevel = LogLevel::kWarn;
        }
        else if (strAppLogLevel == "kError") {
            appLogLevel = LogLevel::kError;
        }
        else if (strAppLogLevel == "kCritical") {
            appLogLevel = LogLevel::kCritical;
        }
        else if (strAppLogLevel == "kOff") {
            appLogLevel = LogLevel::kOff;
        }
        else {
            appLogLevel = LogLevel::kError;
        }
    }
    else {
        std::cout << "log cfg json parse error!" << std::endl;
        return;
    }

    if (bInitFlag) {
        return;
    }

    // Call log manager init to init the application relate info
    HzLogManager::GetInstance()->InitLogging(appId, appDescription, appLogLevel,
                                            outputMode, directoryPath, maxLogFileStoredNum,
                                            maxSizeOfEachLogFile, pureLogFormat);
    updateEnvLoggerSettings();

    if (nullptr == pChangeLevelThread) {
        pChangeLevelThread = new std::thread(LogSetBySocketThread);
    }

    if (isMain) {
        bInitFlag = true;
    }
}


std::shared_ptr<Logger> CreateLogger(std::string ctxId,
                                     std::string ctxDescription,
                                     LogLevel ctxDefLogLevel) noexcept
{
    auto retLogger = HzLogManager::GetInstance()->creatLogger(ctxId, ctxDescription,
                                                   ctxDefLogLevel, iLoggerCount);
    updateEnvLoggerSettings();
    return retLogger;
}


std::shared_ptr<Logger> CreateOperationLogger(
                                    std::string ctxId,
                                    std::string ctxDescription,
                                    LogLevel ctxDefLogLevel
                                    ) noexcept
{
    auto retLogger = HzLogManager::GetInstance()->CreateOperationLogger(ctxId, ctxDescription,
                                                   ctxDefLogLevel, iLoggerCount);
    updateEnvLoggerSettings();
    return retLogger;
}

void InitMcuLogging(
    const std::string appId
) noexcept
{
    HzLogManager::GetInstance()->initMcuLogging(appId);
}

std::shared_ptr<Logger> CreateMcuLogger(
    const std::string appId,
    const std::string ctxId,
    const LogLevel ctxDefLogLevel
) noexcept
{
    auto retLogger = HzLogManager::GetInstance()->createMcuLogger(appId, ctxId, ctxDefLogLevel);
    return retLogger;
}

}
}
}
