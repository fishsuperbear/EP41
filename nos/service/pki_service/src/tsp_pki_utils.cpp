#include "tsp_pki_utils.h"
// #include <sys_ctr.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include "tsp_pki_log.h"

namespace hozon {
namespace netaos {
namespace tsp_pki {

const std::string  APP_PKI_SERVICE_PATH  = "/app/runtime_service/pki_service";

uint64_t TspPkiUtils::GetDataTimeSec() {
    struct timespec time = {0};
    clock_gettime(CLOCK_REALTIME, &time);
    return static_cast<uint64_t>(time.tv_sec);
}

uint64_t TspPkiUtils::GetMgmtTimeSec() {
    struct timespec time = {0};
    // clock_gettime(CLOCK_VIRTUAL, &time);
    return static_cast<uint64_t>(time.tv_sec);
}

std::string TspPkiUtils::GetAppDirectory() {
    // char buf[1024] = {0};
    // if (!getcwd(buf, sizeof(buf))) {
    //     // std::cout << "Can not get working directory.\n";
    //     return "";
    // }
    // std::string wk(buf);
    // if (wk.size() <= 0) {
    //     // std::cout << "Can not get working directory.\n";
    //     return "";
    // }
    // std::cout<<"GetAppDirectory:" <<wk<<std::endl;
    return APP_PKI_SERVICE_PATH;
}

std::string TspPkiUtils::ConvertTime2ReadableStr(uint64_t sec) {
    struct tm timeinfo = {0};
    time_t temp = sec;
    localtime_r(&temp, &timeinfo);
    char time_buf[128] = {0};

    snprintf(time_buf, sizeof(time_buf) - 1, "%04d/%02d/%02d %02d:%02d:%02d", 
                                             timeinfo.tm_year + 1900,
                                             timeinfo.tm_mon + 1,
                                             timeinfo.tm_mday,
                                             timeinfo.tm_hour,
                                             timeinfo.tm_min,
                                             timeinfo.tm_sec);
    return std::string(time_buf);
}

void TspPkiUtils::SetThreadName(std::string name) {
    if (name.size() >= 16) {
        name = name.substr(0, 15);
    }
    pthread_setname_np(pthread_self(), name.c_str());
}

bool TspPkiUtils::IsFileExist(const std::string& file_path) {
    struct stat stat_data;
    if ((stat(file_path.c_str(), &stat_data) == 0) && (S_ISREG(stat_data.st_mode))) {
        return true;
    }
    return false;
}

bool TspPkiUtils::IsDirExist(const std::string& dir_path) {
    struct stat stat_data;
    if ((stat(dir_path.c_str(), &stat_data) == 0) && (S_ISDIR(stat_data.st_mode))) {
        return true;
    }
    return false;
}

bool TspPkiUtils::IsPathExist(const std::string& path) {
    struct stat stat_data;
    if (stat(path.c_str(), &stat_data) == 0) {
        return true;
    }
    return false;
}

bool TspPkiUtils::MakeSurePath(const std::string& path) {

    std::string temp_path;
    // Check if it is a relative path.
    if (!path.empty() && (*(path.begin()) != '/')) {
        // Get working directory.
        char buf[1024] = {0};
        if (!getcwd(buf, sizeof(buf))) {
            PKI_ERROR << "Can not get working directory.";
            return false;
        }
        std::string wk(buf);
        if (wk.size() <= 0) {
            PKI_ERROR << "Can not get working directory.";
            return false;
        }

        if (wk[wk.size() - 1] != '/') {
            wk = wk + '/';
        }

        // Format absolute path.
        temp_path = wk + path;
    }
    else {
        temp_path = path;
    }

    temp_path = NormalizePath(temp_path);

    // Check if path already exsits.
    if (0 != access(temp_path.c_str(), F_OK)) {

        if ((temp_path.size() >= 2) 
            && (temp_path[temp_path.size() - 2] == '.') 
            && (temp_path[temp_path.size() - 1] == '.')) {
            return false;
        }

        std::string parent_path;
        size_t last_slash_pos = temp_path.rfind('/');
        if (last_slash_pos != std::string::npos) {
            parent_path = std::string(temp_path.c_str(), last_slash_pos);

            if (!MakeSurePath(parent_path)) {
                return false;
            }

            if (0 != mkdir(temp_path.c_str(), S_IRWXU)) {
                // std::cout << "Cannot create directory for path: " << temp_path << std::endl;
                return false;
            }
        }
    }

    return true;
}

std::string TspPkiUtils::NormalizePath(const std::string& path) {
    std::string temp = path;
    while (temp.find("/./") != std::string::npos) {
        temp.replace(temp.find("/./"), 3, "/");
    }
    while (temp.find("//") != std::string::npos) {
        temp.replace(temp.find("//"), 2, "/");
    }
    
    // std::cout << "Normalized path: " << temp << std::endl;
    return temp;
}


int64_t TspPkiUtils::GetFileSize(const std::string& file) {
    if(file.empty()) return -2;
	std::ifstream infile(file,std::ios::in);
    if(!infile.is_open()){
        return -1;
        PKI_ERROR <<file << " is not exist.";
    }

    int64_t filesize = 0;
    infile.seekg(0, std::ios::end);
    auto pos = infile.tellg();
    infile.seekg(0, std::ios::beg);
    infile.close();
    filesize = static_cast<int64_t>(pos);
    return filesize;
}


}
}  // namespace datacollect
}  // namespace hozon