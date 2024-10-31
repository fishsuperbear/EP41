#include "diag/ipc/common/ipc_funtions.h"
#include "diag/ipc/common/ipc_def.h"

// void* 转换为 std::vector<uint8_t>
std::vector<uint8_t> VoidPointerToVector(void* data, int size) {
    uint8_t* byteData = static_cast<uint8_t*>(data);
    std::vector<uint8_t> vecData(byteData, byteData + size);
    return vecData;
}

// std::vector<uint8_t> 转换为 void*
void* VectorToVoidPointer(const std::vector<uint8_t>& vecData) {
    void* data = static_cast<void*>(const_cast<uint8_t*>(vecData.data()));
    return data;
}


// char* 转换为 std::vector<uint8_t>
std::vector<uint8_t> CharPointerToVector(const char* data, size_t dataSize) {
    std::vector<uint8_t> vecData;
    if (data != nullptr) {
        const uint8_t* byteData = reinterpret_cast<const uint8_t*>(data);
        vecData.assign(byteData, byteData + dataSize);
    }
    return vecData;
}

bool IPCPathRemove(const std::string &pathName) {
    bool bRet = false;

    if (!doesFileExistWithPrefix(pathName, hozon::netaos::diag::prefix))
    {
        bRet = true;
        return bRet;
    }
    std::string rmCMD = "rm -r  " + pathName + hozon::netaos::diag::prefix + "*";
    if (0 == system(rmCMD.c_str())){
        bRet = true;
    }
    return bRet;
}

bool doesFileExistWithPrefix(const std::string& directoryPath, const std::string& prefix) {
    DIR* dir;
    struct dirent* entry;

    if ((dir = opendir(directoryPath.c_str())) != nullptr) {
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;

            if (filename.rfind(prefix, 0) == 0) {
                closedir(dir);
                return true;
            }
        }

        closedir(dir);
    } else {
        std::cerr << "Failed to open directory: " << directoryPath << std::endl;
    }

    return false;
}