/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: gdb infomation monitor
*/

#include "phm_server/include/common/thread_pool.h"
#include "phm_server/include/common/time_manager.h"
#include "phm_server/include/common/phm_server_config.h"
#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/health_manager/monitor/phm_health_gdb_monitor.h"
#include <dirent.h>
#include <sys/stat.h>
#include <mutex>


namespace hozon {
namespace netaos {
namespace phm_server {

const char FM_GDB_PATH[] = "/opt/usr/col/fm/gdb";
const char CORE_PATH[] = "/opt/usr/data/coredump";

bool m_bInhibitCollectGdb = false;
int m_inhibitCollectGdbTimerFd = -1;
std::mutex m_InhibitCollectGdbMtx;


PhmHealthGdbMonitor::PhmHealthGdbMonitor()
: m_spThreadPool(new ThreadPool(1))
, m_spTimerManager(new TimerManager())
{
    PHMS_INFO << "PhmHealthGdbMonitor::PhmHealthGdbMonitor";
    m_spTimerManager->Init();
    StartInhibitCollectTimer();
}

PhmHealthGdbMonitor::~PhmHealthGdbMonitor()
{
    PHMS_INFO << "PhmHealthGdbMonitor::~PhmHealthGdbMonitor";
    if (m_spTimerManager) {
        m_spTimerManager->DeInit();
        m_spTimerManager = nullptr;
    }

    if (m_spThreadPool) {
        m_spThreadPool->Stop();
        m_spThreadPool = nullptr;
    }
}

void SetInhibitCollectGdb(void* data)
{
    PHMS_INFO << "PhmHealthGdbMonitor::SetInhibitCollectGdb inhibitCollectGdb true";
    std::unique_lock<std::mutex> lck(m_InhibitCollectGdbMtx);
    m_bInhibitCollectGdb = true;
    m_inhibitCollectGdbTimerFd = -1;
}

bool GetInhibitCollectGdb()
{
    std::unique_lock<std::mutex> lck(m_InhibitCollectGdbMtx);
    PHMS_INFO << "PhmHealthGdbMonitor::GetInhibitCollectGdb inhibitCollectGdb:" << m_bInhibitCollectGdb;
    return m_bInhibitCollectGdb;
}

void PhmHealthGdbMonitor::StartInhibitCollectTimer()
{
    const PhmConfigInfo& configInfo = PHMServerConfig::getInstance()->GetPhmConfigInfo();
    PHMS_INFO << "PhmHealthGdbMonitor::StartInhibitCollectTimer SystemCheckTime:" << configInfo.SystemCheckTime;
    if (m_spTimerManager) {
        m_spTimerManager->StartFdTimer(m_inhibitCollectGdbTimerFd,
            configInfo.SystemCheckTime, std::bind(SetInhibitCollectGdb, std::placeholders::_1), nullptr);
    }

    return;
}

void PhmHealthGdbMonitor::CheckGdbDir()
{
    PHMS_INFO << "PhmHealthGdbMonitor::CheckGdbDir";
    if (m_spThreadPool) m_spThreadPool->Commit(std::bind(&PhmHealthGdbMonitor::DoCheckGdbDir, this));
    return;
}

void PhmHealthGdbMonitor::DoCheckGdbDir()
{
    PHMS_INFO << "PhmHealthGdbMonitor::DoCheckGdbDir";
    CheckGdbDirSize();
    CheckGdbInvalidFile();
    return;
}

void PhmHealthGdbMonitor::CheckGdbDirSize()
{
    PHMS_INFO << "PhmHealthGdbMonitor::CheckGdbDirSize";
    struct dirent* pDirent;
    DIR* pDir = opendir(FM_GDB_PATH);
    if (pDir == NULL) {
        PHMS_INFO << "PhmHealthGdbMonitor::CheckGdbDirSize not gdb dir";
        return;
    }

    uint32_t totalSize = 0;
    while ((pDirent = readdir(pDir)) != NULL) {
        std::string file(pDirent->d_name);
        if ("." == file || ".." == file) {
            continue;
        }

        std::string path = std::string(FM_GDB_PATH) + "/" + file;
        struct stat statbuf;
        stat(path.c_str(), &statbuf);

        totalSize += statbuf.st_size;
    }

    closedir(pDir);
    PHMS_INFO << "PhmHealthGdbMonitor::CheckGdbDirSize totalSize: " << totalSize;
    if (totalSize > 1024 * 1024) {
        /* If the directory size exceeds 1 Mb, empty the directory */
        char const* cmd = "rm /opt/usr/col/fm/gdb/*";
        pid_t status = system(cmd);
        if (status == -1) {
            PHMS_INFO << "PhmHealthGdbMonitor::CheckGdbDirSize GdbDir error!";
        }
    }
}

void PhmHealthGdbMonitor::CheckGdbInvalidFile()
{
    PHMS_INFO << "PhmHealthGdbMonitor::CheckGdbInvalidFile";
    struct dirent* pDirent;
    DIR* pDir = opendir(FM_GDB_PATH);
    if (pDir == NULL) {
        PHMS_INFO << "PhmHealthGdbMonitor::CheckGdbInvalidFile not core dir";
        return;
    }

    while ((pDirent = readdir(pDir)) != NULL) {
        std::string file(pDirent->d_name);
        if ("." == file || ".." == file) {
            continue;
        }

        std::string path = std::string(FM_GDB_PATH) + "/" + file;
        struct stat statbuf;
        stat(path.c_str(), &statbuf);
        PHMS_DEBUG << "file:" << path << ",size:" << statbuf.st_size;
        if (0 == statbuf.st_size) {
            remove(path.data());
        }
    }

    closedir(pDir);
    return;
}

void PhmHealthGdbMonitor::StartCollectGdbInfo(const std::string& processName)
{
    PHMS_INFO << "PhmHealthGdbMonitor::StartCollectGdbInfo processName:" << processName;
    if (m_spThreadPool) m_spThreadPool->Commit(std::bind(&PhmHealthGdbMonitor::DoCollectGdbInfo, this, processName));
    return;
}

int PhmHealthGdbMonitor::DoCollectGdbInfo(const std::string& processName)
{
    PHMS_INFO << "PhmHealthGdbMonitor::DoCollectGdbInfo process:" << processName;
    if (!GetInhibitCollectGdb()) {
        return -1;
    }

    if (processName.empty()) {
        PHMS_INFO << "PhmHealthGdbMonitor::DoCollectGdbInfo process empty";
        return -1;
    }

    std::string procNameOfNoProcess = processName;
    const std::string s = "Process";
    procNameOfNoProcess.erase(procNameOfNoProcess.find(s), s.size());

    struct dirent* pDirent;
    DIR* pDir = opendir(CORE_PATH);
    if (pDir == NULL) {
        PHMS_INFO << "PhmHealthGdbMonitor::DoCollectGdbInfo opendir failed";
        return -1;
    }

    uint32_t system_hz = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch()).count()) / 1000000000;
    while ((pDirent = readdir(pDir)) != NULL) {
        std::string gdbFile(pDirent->d_name);
        if ("." == gdbFile || ".." == gdbFile) {
            continue;
        }

        PHMS_DEBUG << "PhmHealthGdbMonitor::DoCollectGdbInfo gdbFile |" << gdbFile << "|";
        size_t pos_1 = gdbFile.find(procNameOfNoProcess);
        if (pos_1 != std::string::npos) {
            size_t pos_2 = gdbFile.find("-");
            pos_1 = 0;
            if (pos_2 != std::string::npos) {
                gdbFile.substr(pos_1, pos_2 - pos_1);
                pos_1 = pos_2 + 1;
                pos_2 = gdbFile.find("-", pos_1);
            }
            else {
                closedir(pDir);
                return -1;
            }

            std::string pidS = gdbFile.substr(pos_1, pos_2 - pos_1);
            if (pos_2 != std::string::npos) {
                gdbFile.substr(pos_1, pos_2 - pos_1);
                pos_1 = pos_2 + 1;
                pos_2 = gdbFile.find("-", pos_1);
            }
            else {
                closedir(pDir);
                return -1;
            }

            std::string tickcountS = gdbFile.substr(pos_1);
            unsigned int tickcountI = 0;
            std::stringstream ss;
            ss << tickcountS;
            ss >> tickcountI;
            PHMS_DEBUG << "------ match gdb file: " << gdbFile.c_str() << ", tickcount: " << tickcountI << ", curr tickcount: " << system_hz;
            if (system_hz - tickcountI > 150) {
                continue;
            }

            char cmd_format[550] = {0};
            memset(cmd_format, 0, sizeof(cmd_format));
            snprintf(cmd_format, sizeof(cmd_format),
                "gdb --eval-command=\"thread apply all bt\" --batch /app/runtime_service/%s/bin/%s /opt/usr/data/coredump/%s > /opt/usr/col/fm/gdb/core_%s_%s_%d.gdb.log.tmp;" \
                "mv /opt/usr/col/fm/gdb/core_%s_%s_%d.gdb.log.tmp /opt/usr/col/fm/gdb/core_%s_%s_%d.gdb.log",
                procNameOfNoProcess.c_str(), procNameOfNoProcess.c_str(), gdbFile.c_str(),
                procNameOfNoProcess.c_str(), pidS.c_str(), tickcountI,
                procNameOfNoProcess.c_str(), pidS.c_str(), tickcountI,
                procNameOfNoProcess.c_str(), pidS.c_str(), tickcountI);

            pid_t status = system(cmd_format);
            if (status == -1) {
                PHMS_ERROR << "PhmHealthGdbMonitor system call failed!";
            }
        }
    }

    closedir(pDir);
    // PHMS_INFO << "PhmHealthGdbMonitor::DoCollectGdbInfo end";
    return 0;
}


}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
