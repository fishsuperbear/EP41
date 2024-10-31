#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>
#include <time.h>
#include <map>
#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/common/phm_server_utils.h"

namespace hozon {
namespace netaos {
namespace phm_server {

uint64_t PHMUtils::GetTimeMicroSec() {
    struct timespec time = {0};
    clock_gettime(CLOCK_REALTIME, &time);
    uint64_t micro_secs = time.tv_sec * 1000000 + time.tv_nsec / 1000;

    return micro_secs;
}

uint32_t PHMUtils::GetTimeSec() {
    struct timespec time = {0};
    clock_gettime(CLOCK_REALTIME, &time);
    return static_cast<uint32_t>(time.tv_sec);
}

uint32_t PHMUtils::GetTimeMicroSecInSec() {
    struct timespec time = {0};
    clock_gettime(CLOCK_REALTIME, &time);
    return static_cast<uint32_t>(time.tv_nsec / 1000);
}

Timestamp PHMUtils::GetTimestamp() {
   struct timespec time = {0};
    clock_gettime(CLOCK_REALTIME, &time);

    Timestamp timestamp {static_cast<uint32_t>(time.tv_nsec), static_cast<uint32_t>(time.tv_sec)};
    return timestamp;
}

void PHMUtils::SetThreadName(std::string name) {

    if (name.size() > 16) {
        name = name.substr(name.size() - 15, 15);
    }
    pthread_setname_np(pthread_self(), name.c_str());
}

bool PHMUtils::IsFileExist(const std::string& file_path) {
    struct stat stat_data;
    if ((stat(file_path.c_str(), &stat_data) == 0) && (S_ISREG(stat_data.st_mode))) {
        return true;
    }
    return false;
}

bool PHMUtils::IsDirExist(const std::string& dir_path) {
    struct stat stat_data;
    if ((stat(dir_path.c_str(), &stat_data) == 0) && (S_ISDIR(stat_data.st_mode))) {
        return true;
    }
    return false;
}

bool PHMUtils::IsPathExist(const std::string& path) {
    struct stat stat_data;
    if (stat(path.c_str(), &stat_data) == 0) {
        return true;
    }
    return false;
}

bool PHMUtils::MakeSurePath(const std::string& path) {

    // Check if path already exsits.
    if (0 != access(path.c_str(), F_OK)) {
        std::string parent_path;
        std::size_t last_slash_pos = path.rfind('/');
        if (last_slash_pos != std::string::npos) {
            parent_path = std::string(path.c_str(), last_slash_pos);
            if (0 != mkdir(parent_path.c_str(), S_IRWXU)) {
                return false;
            }
        }
    }

    return true;
}

bool PHMUtils::CopyFile(const std::string& from_path, const std::string& to_path)
{
    std::ifstream input(from_path, std::ios::binary);
    if (!input.good()) {
        return false;
    }

    std::ofstream output(to_path, std::ios::binary);
    if (!output.good()) {
        input.close();
        return false;
    }

    output << input.rdbuf();
    input.close();
    output.close();
    return true;
}

bool PHMUtils::RenameFile(const std::string& old_path, const std::string& new_path) {

    struct stat buf;
    const int32_t ret = lstat(old_path.c_str(), &buf);
    if (ret == -1) {
        return false;
    }
    if (S_ISREG(buf.st_mode) == 0) {
        return false;
    }
    return (rename(old_path.c_str(), new_path.c_str()) == 0);
}

bool PHMUtils::RemoveFile(std::string file_path) {

    struct stat buf;
    const int32_t ret = lstat(file_path.c_str(), &buf);
    if (ret == -1) {
        return false;
    }
    if (S_ISREG(buf.st_mode) == 0) {
        return false;
    }
    return (remove(file_path.c_str()) == 0);
}

std::string PHMUtils::GetFileName(std::string path) {
    std::string file_name;

    size_t slash_pos = path.rfind("/");
    if ((slash_pos != std::string::npos) && (slash_pos < (path.size() - 1))) {
        file_name = path.substr(slash_pos + 1, path.size() - slash_pos - 1);
    }

    return file_name;
}

std::shared_ptr<std::vector<uint8_t>> PHMUtils::ReadFile(std::string file_path) {

    auto file_buf = std::make_shared<std::vector<uint8_t>>();

    std::unique_ptr<FILE, void (*)(FILE*)> file(fopen(file_path.c_str(), "rb"), [](FILE* f) {
        if (f) {
            fclose(f);
        }
    });

    if (file) {
        fseek(file.get(), 0, SEEK_END);
        uint32_t file_size = ftell(file.get());
        fseek(file.get(), 0, SEEK_SET);

        if (file_size > 0) {
            file_buf->resize(file_size);
            if (file_size != fread(file_buf->data(), 1, file_size, file.get())) {
                file_buf->clear();
                file_buf = nullptr;
            }
        }
    }

    if (file_buf->size() <= 0) {
        return file_buf;
    }

    return file_buf;
}

bool PHMUtils::WriteFile(std::string file_path, std::shared_ptr<std::vector<uint8_t>> buf) {

    bool ret = false;
    std::unique_ptr<FILE, void (*)(FILE*)> file(::fopen(file_path.c_str(), "wb+"), [](FILE* f) { ::fclose(f); });

    if (file && (1 == ::fwrite(buf->data(), buf->size(), 1, file.get()))) {
        ret = true;
    }

    return ret;
}
std::string PHMUtils::FormatTimeStrForFileName(time_t unix_time) {
    struct tm timeinfo = {0};
    localtime_r(&unix_time, &timeinfo);
    char time_buf[128] = {0};

    snprintf(time_buf, sizeof(time_buf) - 1, "%04d%02d%02d-%02d%02d%02d",
                                             timeinfo.tm_year + 1900,
                                             timeinfo.tm_mon + 1,
                                             timeinfo.tm_mday,
                                             timeinfo.tm_hour,
                                             timeinfo.tm_min,
                                             timeinfo.tm_sec);
    return std::string(time_buf);
}

uint64_t PHMUtils::GetCurrentTime()
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME,&ts);
    return static_cast<uint64_t>(ts.tv_sec*1000000000+ts.tv_nsec);
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
