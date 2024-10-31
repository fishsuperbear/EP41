#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>

#include "adf/include/internal_log.h"
#include "adf/include/node_resource_limit.h"

#define DO(statement)      \
    if ((statement) < 0) { \
        return -1;         \
    }

#define DO_OR_ERROR(statement, errortxt) \
    if ((statement) < 0) {               \
        ADF_LOG_ERROR << errortxt;       \
        return -1;                       \
    }

namespace hozon {
namespace netaos {
namespace adf {
int32_t NodeResourceLimit::LimitCpu(const std::string& group_name, uint32_t percentage) {
    std::string dir = "/sys/fs/cgroup/cpu/" + group_name;

    DO(CreateDir(dir));
    DO(WriteTrunc(dir + "/cpu.cfs_quota_us", std::to_string(percentage * 1000)));

    pid_t pid = getpid();
    DO(WriteTrunc(dir + "/tasks", std::to_string(pid)));

    return 0;
}

int32_t NodeResourceLimit::LimitMem(const std::string& group_name, uint32_t memory_mb) {
    std::string dir = "/sys/fs/cgroup/memory/" + group_name;

    DO(CreateDir(dir));
    DO(WriteTrunc(dir + "/memory.limit_in_bytes", std::to_string(memory_mb * 1024 * 1024)));
    DO(WriteTrunc(dir + "/memory.memsw.limit_in_bytes", std::to_string(memory_mb * 1024 * 1024)));

    pid_t pid = getpid();
    DO(WriteTrunc(dir + "/tasks", std::to_string(pid)));

    return 0;
}

int32_t NodeResourceLimit::CreateDir(const std::string& dir) {
    struct stat st;

    if (stat(dir.c_str(), &st) == 0) {
        if (!S_ISDIR(st.st_mode)) {
            std::cout << dir << " already exists and is not folder.\n";
            return -1;
        } else {
            return 0;
        }
    }

    DO_OR_ERROR(mkdir(dir.c_str(), 0755), "Fail to create cgroup dir " << dir);
    return 0;
}

int32_t NodeResourceLimit::WriteTrunc(const std::string& file, const std::string& content) {
    std::fstream fs;
    fs.open(file, std::ios::trunc | std::ios::out);
    DO_OR_ERROR((fs.is_open() ? 0 : -1), "Fail to open " << file);
    fs << content;
    fs.close();

    return 0;
}
}  // namespace adf
}  // namespace netaos
}  // namespace hozon