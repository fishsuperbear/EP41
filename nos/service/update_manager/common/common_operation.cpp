/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: data definition
 */
#include <sys/stat.h>
#include "update_manager/common/common_operation.h"
#include "update_manager/log/update_manager_logger.h"
#include <unzipper.h>

namespace hozon {
namespace netaos {
namespace update {

bool PathExists(const std::string &pathName)
{
    std::ifstream file(pathName);
    return file.good();
}

bool PathClear(const std::string &pathName)
{
    bool bRet = false;

    if (!PathExists(pathName))
    {
        bRet = true;
        return bRet;
    }
    std::string rmCMD = "rm -rf  " + pathName + "*";
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
    UPDATE_LOG_D("PathCreate path: %s", pathName.c_str());
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
    UPDATE_LOG_D("FileRecovery file: %s to path: %s.", file.c_str(), pathName.c_str());
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
        UPDATE_LOG_E("zipFileName: %s is not existed!~", zipFileName.c_str());
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

bool FileMount(const std::string &pathName, const std::string &mountPath)
{
    UPDATE_LOG_D("FileMount file: %s to path: %s.", pathName.c_str(), mountPath.c_str());
    bool bRet = false;

    if (!PathExists(pathName))
    {
        UPDATE_LOG_W("file %s not Exists", pathName.c_str());
        return bRet;
    }

    std::string mountCMD = "mount -o loop " + pathName + "  " + mountPath;
    if (0 == system(mountCMD.c_str())){
        bRet = true;
    }
    return bRet;
}

bool FileUmount(const std::string &umountPath)
{
    UPDATE_LOG_D("FileUmount path: %s.", umountPath.c_str());
    bool bRet = false;

    if (!PathExists(umountPath))
    {
        UPDATE_LOG_W("file %s not Exists", umountPath.c_str());
        return bRet;
    }

    std::string umountCMD = "umount -l " + umountPath;
    if (0 == system(umountCMD.c_str())){
        bRet = true;
    }
    return bRet;
}

bool SystemSync()
{
    bool bRet = false;
    std::string syncCMD = "sync";
    if (0 == system(syncCMD.c_str())){
        UPDATE_LOG_D("sync done");
        bRet = true;
    }
    return bRet;
}

bool GetAbsolutePath(const std::string& relativePath, std::string& absolutePath)
{
    UPDATE_LOG_D("input path: %s.", relativePath.c_str());

    char resolvedPath[4096];
    char* result = realpath(relativePath.c_str(), resolvedPath);
    
    if (result) {
        absolutePath = std::string(resolvedPath);
        if (!PathExists(absolutePath))
        {
            UPDATE_LOG_W("file %s not Exists", absolutePath.c_str());
            return false;
        } else {
            UPDATE_LOG_D("absolutePath is : %s", absolutePath.c_str());
            return true;
        }
    } else {
        absolutePath = "";
        return false;
    }
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

int32_t getFileSize(const std::string &FilePath)
{
    struct stat st;
    stat(FilePath.c_str(), &st);
    return static_cast<uint32_t>(st.st_size);
}

int16_t createFile(const std::string& filePath) 
{
    std::ifstream file(filePath);
    if (file.good()) {
        file.close();
        return 0;
    }

    std::ofstream newFile(filePath);
    if (!newFile.is_open()) {
        UPDATE_LOG_E("Error: Unable to create file.");
        return -1;
    }

    newFile.close();
    return 0;
}

int16_t writeToFile(const std::string& filePath, const std::string& content) 
{
    std::ofstream file(filePath, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        UPDATE_LOG_E("Error: Unable to open file for writing.");
        return -1;
    }

    file << content;
    file.close();
    return 0;
}

int16_t readFile(const std::string& filePath, std::string& content) 
{
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for reading." << std::endl;
        return -1; // Failed to open file for reading
    }

    content.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return 0; // Content read successfully
}


}  // namespace update
}  // namespace netaos
}  // namespace hozon
