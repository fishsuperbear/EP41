/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: file_utils.cpp
 * @Date: 2023/08/15
 * @Author: cheng
 * @Desc: --
 */

#include "utils/include/path_utils.h"
#include "utils/include/dc_logger.hpp"
//#include <sys_ctr.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <filesystem>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <map>
#include <filesystem>
#include <regex>

namespace hozon {
namespace netaos {
namespace dc {


const char pathSpliter = '/';
static uint8_t s_utils_running_mode = 1;

//
//std::string PathUtils::getAppDirectory() {
//    char buf[1024] = {0};
//    if (!getcwd(buf, sizeof(buf))) {
//        DC_SERVER_LOG_ERROR << "Can not get working directory.\n";
//        return "";
//    }
//    std::string wk(buf);
//    if (wk.size() <= 0) {
//        DC_SERVER_LOG_ERROR << "Can not get working directory.\n";
//        return "";
//    }
//
//    if (wk[wk.size() - 1] != '/') {
//        wk = wk + '/';
//    }
//
//    std::string temp_bin_dir = wk + "bin";
//
//    struct stat bin_dir_stat;
//    if (stat(temp_bin_dir.c_str(), &bin_dir_stat) == 0) {
//        if (S_ISDIR(bin_dir_stat.st_mode)) {
//            return wk;
//        }
//    }
//
//    return APP_DIR_DATA_COLLECT;
//}

bool PathUtils::isFileExist(const std::string& file_path) {
    struct stat stat_data;
    if ((stat(file_path.c_str(), &stat_data) == 0) && (S_ISREG(stat_data.st_mode))) {
        return true;
    }
    return false;
}

bool PathUtils::isDirExist(const std::string& dir_path) {
    struct stat stat_data;
    if ((stat(dir_path.c_str(), &stat_data) == 0) && (S_ISDIR(stat_data.st_mode))) {
        return true;
    }
    return false;
}

bool PathUtils::isPathExist(const std::string& path) {
    struct stat stat_data;
    if (stat(path.c_str(), &stat_data) == 0) {
        return true;
    }
    return false;
}

bool PathUtils::createFoldersIfNotExists(const std::string& path) {
    // Check if path already exsits.
    if (0 != access(path.c_str(), F_OK)) {
        std::string parent_path;
        std::size_t last_slash_pos = path.rfind(pathSpliter);
        if (last_slash_pos != std::string::npos) {
            parent_path = std::string(path.c_str(), last_slash_pos);
            if (!createFoldersIfNotExists(parent_path)) {
                return false;
            }
            if (path.back() == pathSpliter) {
                return true;
            }
            if (0 != mkdir(path.c_str(), S_IRWXU)) {
                DC_SERVER_LOG_ERROR << "Cannot create directory for path: " << path;
                return false;
            }
        }
    }

    return true;
}

bool PathUtils::removeOldFile(std::string dir_path, uint32_t max_file) {
    struct dirent* ptr = NULL;
    DIR* dir = opendir(dir_path.c_str());
    if (dir == NULL) {
        return false;
    }

    struct TimeSpecComp {
        bool operator()(const timespec& lhs, const timespec& rhs) const {
            if (lhs.tv_sec == rhs.tv_sec) {

                return (lhs.tv_nsec < rhs.tv_nsec);
            } else {
                return (lhs.tv_sec < rhs.tv_sec);
            }
        }
    };
    std::map<struct timespec, std::string, TimeSpecComp> files;

    ptr = readdir(dir);
    while (ptr != NULL) {
        if (ptr->d_name[0] != '.') {
            const std::string child_path = std::string(dir_path) + "/" + std::string(ptr->d_name);
            struct stat buf;
            const int32_t ret = lstat(child_path.c_str(), &buf);
            if (ret == -1) {
                continue;
            }
            if (S_ISREG(buf.st_mode) != 0) {
                files[buf.st_mtim] = child_path;
            } else if (S_ISDIR(buf.st_mode) != 0) {
                // Recursive is not supported.
            } else {
            }
        }
        ptr = readdir(dir);
    }

    closedir(dir);

    while (files.size() >= max_file) {
        std::string file_path = files.begin()->second;

        if (remove(file_path.c_str()) == 0) {
            DC_SERVER_LOG_ERROR << "Remove oldest file successfully :  " << file_path;
        } else {
            DC_SERVER_LOG_ERROR << "Remove oldest file failed :  " << file_path;
            return false;
        }

        files.erase(files.begin());
    }

    return true;
}

bool PathUtils::removeFile(std::string file_path) {

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

bool PathUtils::removeFilesInFolder(std::string folder_path) {
    if (!isDirExist(folder_path)) {
        return false;
    }
    if (!std::filesystem::is_empty(folder_path)) {
        // 遍历文件夹下的所有文件和子目录
        for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
            // 判断是文件
            if (std::filesystem::is_regular_file(entry)) {
                // 删除文件
                std::filesystem::remove(entry.path());
            }
        }
    }
    return true;
}

bool PathUtils::renameFile(std::string old_file_path, std::string new_file_path) {
    if (!isFileExist(old_file_path)) {
        DC_SERVER_LOG_ERROR << "the file not exists for renameFile:" + old_file_path;
        return false;
    }
    std::filesystem::path old_path = old_file_path;
    std::filesystem::path new_path = new_file_path;
    std::filesystem::rename(old_path, new_path);
    return true;
}

bool PathUtils::removeFolder(std::string folder_path) {
    if (!isDirExist(folder_path)) {
        DC_SERVER_LOG_ERROR << "the folder not exists for removeFolder:" + folder_path;
        return false;
    }
    std::filesystem::remove_all(folder_path);
    return true;
}

bool PathUtils::clearFolder(std::string path) {
    if (!isDirExist(path)) {
        DC_SERVER_LOG_ERROR << "the folder not exists for getFiles:" + path;
        return false;
    }
    bool result = true;
    for (const auto & entry : std::filesystem::directory_iterator(path)) {
        if (std::filesystem::is_regular_file(entry)) {
            result = removeFile(entry.path().string());
        } else if (std::filesystem::is_directory(entry)) {
            result = removeFolder(entry.path().string());
        }
    }
    return result;
}

bool PathUtils::fastRemoveFolder(std::string folder_path, std::string fast_name) {
    if (!isDirExist(folder_path)) {
        DC_SERVER_LOG_ERROR << "the folder not exists for removeFolder:" + folder_path;
        return false;
    }
    if ((folder_path.find(fast_name) == 0) && (folder_path.size() > fast_name.size())) {
        std::filesystem::remove_all(folder_path);
        return true;
    } else {
        DC_SERVER_LOG_DEBUG << "the folder not fast for removeFolder:" + folder_path;
        return false;
    }
}

std::string PathUtils::getFileName(std::string path) {
    std::string file_name;

    size_t slash_pos = path.rfind("/");
    if ((slash_pos != std::string::npos) && (slash_pos < (path.size() - 1))) {
        file_name = path.substr(slash_pos + 1, path.size() - slash_pos - 1);
    }

    return file_name;
}

std::string PathUtils::getFolderName(std::string path) {
    std::string folder_name = "";
    size_t slash_pos = path.rfind('/');
	if (slash_pos != std::string::npos) {
		folder_name = path.substr(0, slash_pos);
	}
	return folder_name;
}

bool PathUtils::readFile(std::string file_path, std::vector<char>& content) {

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
            content.resize(file_size);
            if (file_size != fread(content.data(), 1, file_size, file.get())) {
                DC_SERVER_LOG_ERROR << "Read file failed. file path: " << file_path;
                content.clear();
            }
        }
    }

    if (content.size() <= 0) {
        return false;
    }

    return true;
}

std::shared_ptr<std::vector<uint8_t>> PathUtils::readFile(std::string file_path) {

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
                DC_SERVER_LOG_ERROR << "Read file failed. file path: " << file_path;
                file_buf->clear();
            }
        }
    }

    if (file_buf->size() <= 0) {
        return file_buf;
    }

    return file_buf;
}

bool PathUtils::writeFile(std::string file_path, std::shared_ptr<std::vector<uint8_t>> buf) {

    bool ret = false;
    std::unique_ptr<FILE, void (*)(FILE*)> file(::fopen(file_path.c_str(), "wb+"), [](FILE* f) { ::fclose(f); });

    if (file && (1 == ::fwrite(buf->data(), buf->size(), 1, file.get()))) {

        ret = true;
    }

    return ret;
}

bool PathUtils::writeFile(std::string file_path, std::string& str) {
    bool ret = false;
    std::unique_ptr<FILE, void (*)(FILE*)> file(::fopen(file_path.c_str(), "wb+"), [](FILE* f) { ::fclose(f); });
    if (file && (1 == ::fwrite(str.data(), str.size(), 1, file.get()))) {
        ret = true;
    }
    return ret;
}

bool PathUtils::getFiles(std::string path, std::string filePattern, bool searchSubPath, std::vector<std::string>& files) {
    if (!isDirExist(path)) {
        DC_SERVER_LOG_ERROR << "the folder not exists for getFiles:" + path;
        return false;
    }
    bool result = true;
    for (const auto & entry : std::filesystem::directory_iterator(path)) {
        if (std::filesystem::is_regular_file(entry)) {
            std::regex  expression(filePattern);
            if (std::regex_match (entry.path().filename().string(), expression)) {
                files.emplace_back(entry.path());
            }
            continue;
        }
        if (searchSubPath) {
            if (!getFiles(entry.path().string(), filePattern, searchSubPath, files)){
                result = false;
            }
        }
    }
    return result;
}

bool PathUtils::getFiles(std::string path, std::vector<std::string>& files, bool searchSubPath) {
    if (!isDirExist(path)) {
        DC_SERVER_LOG_ERROR << "the folder not exists for getFiles:" + path;
        return false;
    }
    bool result = true;
    for (const auto & entry : std::filesystem::directory_iterator(path)) {
        if (std::filesystem::is_regular_file(entry)) {
            files.emplace_back(entry.path());
            continue;
        }
        if (searchSubPath) {
            if (!getFiles(entry.path().string(), files)) {
                result = false;
            }
        }
    }
    return result;
}

std::string PathUtils::getFilePath(std::string folder_path, std::string file_name) {
    std::string file_path;
    if (folder_path.back() == '/') {
        file_path = folder_path + file_name;
    } else {
        file_path = folder_path + "/" + file_name;
    }
    return file_path;
}

//bool getFiles(std::string path, std::vector<std::string>& files )
//{
//    long   hFile   =   0;
//    struct _finddata_t fileinfo;
//    std::string p;
//    if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)
//    {
//        do
//        {
//            //如果是目录,迭代之
//            //如果不是,加入列表
//            if((fileinfo.attrib &  _A_SUBDIR))
//            {
//                if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
//                    getFiles( p.assign(path).append("\\").append(fileinfo.name), files );
//            }
//            else
//            {
//                files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
//            }
//        }while(_findnext(hFile, &fileinfo)  == 0);
//        _findclose(hFile);
//    }
//    for (auto const& it : std::filesystem::directory_iterator{path}) {
//        if (it.exists() && it.is_regular_file()) {
//            files.emplace_back(std::filesystem::absolute(it).string());
//        }
//    }
//    return true;
//}

uint8_t PathUtils::runningMode() {
    return s_utils_running_mode;
}

void PathUtils::setRunningMode(uint8_t running_mode) {
    s_utils_running_mode = running_mode;
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
