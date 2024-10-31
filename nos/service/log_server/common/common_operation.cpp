#include "log_server/common/common_operation.h"
#include "log_server/log/log_server_logger.h"

namespace hozon {
namespace netaos {
namespace logserver {

bool PathExists(const std::string &pathName)
{
    struct stat buffer;
    return (::stat(pathName.c_str(), &buffer) == 0);
}

bool PathClear(const std::string &pathName)
{
    bool bRet = false;

    if (!PathExists(pathName))
    {
        bRet = true;
        return bRet;
    }
    std::string rmCMD = "rm -r  " + pathName + "/*";
    if (0 == system(rmCMD.c_str())){
        bRet = true;
    }
    return bRet;
}

bool PathRemove(const std::string &pathName)
{
    bool bRet = false;

    if (!PathExists(pathName))
    {
        bRet = true;
        return bRet;
    }
    std::string rmCMD = "rm -r  " + pathName;
    if (0 == system(rmCMD.c_str())){
        bRet = true;
    }
    return bRet;
}

bool PathCreate(const std::string &pathName)
{
    LOG_SERVER_DEBUG << "PathCreate path: " << pathName;
    bool bRet = false;

    if (PathExists(pathName))
    {
        bRet = true;
        return bRet;
    }
    std::string createPathCMD = "mkdir -p  " + pathName;
    if (0 == system(createPathCMD.c_str())){
        bRet = true;
    }
    return bRet;
}

bool FileRecovery(const std::string &file, const std::string &pathName)
{
    LOG_SERVER_DEBUG << "FileRecovery file: " << file << " ,to path: " << pathName;
    bool bRet = false;

    if (!PathExists(file))
    {
        return bRet;
    }

    PathRemove(pathName);
    PathCreate(pathName);

    std::string createPathCMD = "cp  " + file + "  " + pathName;
    if (0 == system(createPathCMD.c_str())){
        bRet = true;
    }
    return bRet;
}

bool UnzipFile(const std::string &zipFileName, const std::string &unzipPath)
{
    bool bRet = false;
    if (0 != access(zipFileName.c_str(), F_OK)) {
        // zip file is not existed.
        LOG_SERVER_ERROR << "zipFileName: " << zipFileName << " is not existed!~";
        return bRet;
    }

    if (!PathExists(unzipPath))
    {
        PathCreate(unzipPath);
    }
    auto pos = zipFileName.find(".zip");

    if (pos == std::string::npos)
    {
        bRet = false;
        return bRet;
    }

    zipper::Unzipper unzipFile(zipFileName);
    unzipFile.extract(unzipPath);
    unzipFile.close();

    return bRet;
}

std::string getFileName(const std::string &FilePath)
{
    if (!PathExists(FilePath))
    {
        return "";
    }
    int pos=FilePath.find_last_of('/');
	std::string s(FilePath.substr(pos+1));

    return s;
}

std::uint32_t getFileSize(const std::string &FilePath)
{
    struct stat st;
    stat(FilePath.c_str(), &st);
    return static_cast<uint32_t>(st.st_size);
}


}  // namespace logserver
}  // namespace netaos
}  // namespace hozon
