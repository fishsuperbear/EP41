// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)

#pragma once

#ifndef SPDLOG_HEADER_ONLY
#    include <spdlog/sinks/hz_rotating_file_sink.h>
#endif

#include <spdlog/common.h>

#include <spdlog/details/file_helper.h>
#include <spdlog/details/null_mutex.h>
#include <spdlog/fmt/fmt.h>

#include <cerrno>
#include <chrono>
#include <ctime>
#include <mutex>
#include <string>
#include <sstream>
#include <tuple>

#include <zipper.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <map>

#include <regex>
#include "zmq_ipc/proto/log_server.pb.h"

namespace spdlog {
namespace sinks {

#define     MAX_LOG_INDEX       (9999)

#ifdef BUILD_LOG_SERVER_ENABLE
#undef BUILD_LOG_SERVER_ENABLE
#endif

namespace
{
    const std::string compress_log_service_name = "tcp://localhost:15778";
}

template<typename Mutex>
SPDLOG_INLINE hz_rotating_file_sink<Mutex>::hz_rotating_file_sink(
    filename_t base_file_path, filename_t file_base_name, std::size_t max_size, std::size_t max_files, bool rotate_on_open, const file_event_handlers &event_handlers)
    : file_path_(base_file_path)
    , file_base_name_(std::move(file_base_name))
    , current_writting_filename_(file_base_name_ + ".log")
    , max_size_(max_size)
    , max_files_(max_files)
    , file_helper_{event_handlers}
    , rotate_on_open_(rotate_on_open)
{
#ifdef BUILD_LOG_SERVER_ENABLE
    client_ = std::make_unique<hozon::netaos::zmqipc::ZmqIpcClient>();
    client_->Init(compress_log_service_name);
#endif
    if (max_size == 0)
    {
        throw_spdlog_ex("rotating sink constructor: max_size arg cannot be zero");
    }

    if (max_files > 200000)
    {
        throw_spdlog_ex("rotating sink constructor: max_files arg cannot exceed 200000");
    }
    log_file_create_();

}

template<typename Mutex>
SPDLOG_INLINE hz_rotating_file_sink<Mutex>::~hz_rotating_file_sink()
{
    // std::cout << "~hz_rotating_file_sink !!!" << std::endl;
#ifdef BUILD_LOG_SERVER_ENABLE
    client_->Deinit();
#endif
}

template<typename Mutex>
SPDLOG_INLINE void hz_rotating_file_sink<Mutex>::log_file_create_()
{
    file_helper_.open(file_path_ + count_file_name_());
    current_size_ = file_helper_.size(); // expensive. called only once

    std::map<std::uint32_t, std::string> log_files = current_files_();
    remove_old_files_(log_files);
}


template<typename Mutex>
SPDLOG_INLINE filename_t hz_rotating_file_sink<Mutex>::filename()
{
    std::lock_guard<Mutex> lock(base_sink<Mutex>::mutex_);
    return file_helper_.filename();
}

template<typename Mutex>
SPDLOG_INLINE void hz_rotating_file_sink<Mutex>::sink_it_(const details::log_msg &msg)
{
    memory_buf_t formatted;
    base_sink<Mutex>::formatter_->format(msg, formatted);

    // if (!details::os::path_exists(file_path_ + current_writting_filename_)){
    //     log_file_create_();
    // }

    file_helper_.write(formatted);
    current_size_ += formatted.size();

    // rotate if the new estimated file size exceeds max size.
    // rotate only if the real size > 0 to better deal with full disk (see issue #2261).
    // we only check the real size when new_size > max_size_ because it is relatively expensive.
    if (current_size_ > max_size_) {
        file_helper_.flush();
        rotate_();
    }
}

template<typename Mutex>
SPDLOG_INLINE void hz_rotating_file_sink<Mutex>::flush_()
{
    // if (!details::os::path_exists(file_path_ + current_writting_filename_)){
    //     log_file_create_();
    // }
    file_helper_.flush();
}

template<typename Mutex>
SPDLOG_INLINE void hz_rotating_file_sink<Mutex>::compress_file_(const filename_t &filename, const filename_t &basename)
{
#ifdef BUILD_LOG_SERVER_ENABLE
    try {
        ZipperInfo info{};
        // 需要拿到绝对路径, 因为两个进程的"./"绝对路径不同
        info.set_file_path(GetAbsolutePath(file_path_));
        info.set_file_base_name(file_base_name_);
        info.set_filename(filename);
        info.set_basename(basename);

        std::string serializedData = info.SerializeAsString();
        errno = 0;
        client_->RequestAndForget(serializedData);
        if (errno == 11) {
            errno = 0;

            zipper::Zipper zipfile(file_path_ + "tmp_" + file_base_name_ + ".zip");
            zipfile.add(filename);
            zipfile.close();
        }
    }
    catch (const zmq::error_t& ex) {
        zipper::Zipper zipfile(file_path_ + "tmp_" + file_base_name_ + ".zip");
        zipfile.add(filename);
        zipfile.close();
    }
#else
    zipper::Zipper zipfile(file_path_ + "tmp_" + file_base_name_ + ".zip");
    zipfile.add(filename);
    zipfile.close();
#endif
}

template<typename Mutex>
SPDLOG_INLINE std::string hz_rotating_file_sink<Mutex>::GetAbsolutePath(const std::string& relativePath)
{
    // 如果输入的是绝对路径，则直接返回
    if (!relativePath.empty() && relativePath.front() == '/') {
        return relativePath;
    }

    std::string absolutePath = "";  // 初始为空

    // 获取当前工作目录的绝对路径
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != nullptr)
    {
        absolutePath = cwd;
    }
    else
    {
        std::cerr << "Failed to get current working directory." << std::endl;
        return "";  // 获取当前工作目录失败，返回空字符串
    }

    // 将相对路径附加到绝对路径后面
    if (!relativePath.empty() && relativePath != "./")
    {
        absolutePath += "/" + relativePath;
    }

    if (absolutePath.back() != '/')
    {
        absolutePath += "/";  // 如果绝对路径最后没有斜杠，则添加斜杠
    }

    return absolutePath;
}


template<typename Mutex>
SPDLOG_INLINE void hz_rotating_file_sink<Mutex>::rotate_()
{
    using details::os::filename_to_str;
    using details::os::path_exists;

    file_helper_.close();

    if (0 == max_files_)
    {
        file_helper_.reopen(true);
        return;
    }

    filename_t current_file = file_path_ + current_writting_filename_;
    if (details::os::path_exists(current_file)){
        filename_t file_name, ext;
        std::tie(file_name, ext) = details::file_helper::split_by_extension(current_writting_filename_);

        filename_t file_to_cmpr = current_writting_filename_;
        filename_t bak_file_name = file_name + ext + "_" ;
        rename_file_(file_path_ + current_writting_filename_, file_path_ + bak_file_name);
        #ifdef BUILD_LOG_SERVER_ENABLE

            filename_t basename, ext2;
            std::tie(basename, ext2) = details::file_helper::split_by_extension(bak_file_name);

            if (details::os::path_exists(file_path_ + "tmp_" + file_base_name_ + ".zip")){
                (void)details::os::remove(file_path_ + "tmp_" + file_base_name_ + ".zip");
            }
            filename_t cmpr_file = GetAbsolutePath(file_path_) + bak_file_name;
            compress_file_(cmpr_file, basename);

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        #else
            // 这里需要值传递，传递引用会导致bak_file_name被销毁，而线程还可能未执行，会产生未初始化的问题
            std::thread([bak_file_name, this](){
                filename_t basename, ext3;
                std::tie(basename, ext3) = details::file_helper::split_by_extension(bak_file_name);

                if (details::os::path_exists(file_path_ + "tmp_" + file_base_name_ + ".zip")){
                    (void)details::os::remove(file_path_ + "tmp_" + file_base_name_ + ".zip");
                }
                filename_t cmpr_file = GetAbsolutePath(file_path_) + bak_file_name;
                compress_file_(cmpr_file, basename);

                rename_file_(file_path_ + "tmp_" + file_base_name_ + ".zip", file_path_ + basename + ".zip");
                (void)details::os::remove(cmpr_file);
            }).detach();
        #endif
    }
    log_file_create_();
}

template<typename Mutex>
SPDLOG_INLINE std::string hz_rotating_file_sink<Mutex>::extractAppFromFilename(const std::string& filename) {
    std::regex pattern(R"((.*)_\d{4}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.[^.]+)");
    std::smatch matches;

    if (std::regex_search(filename, matches, pattern)) {
        std::string app = matches[1].str();
        return app;
    }

    return "";
}

template<typename Mutex>
SPDLOG_INLINE std::map<std::uint32_t, std::string> hz_rotating_file_sink<Mutex>::current_files_()
{
    std::map<std::uint32_t, std::string> log_files;

    /*Get the log file list...*/
    DIR *pDir;
    struct dirent* ptr;

    if(!(pDir = opendir(file_path_.c_str())))
    {
        //std::cout << "current_files_, file_path_ doesn't Exist!  file_path_.c_str():" << file_path_.c_str() << std::endl;
        return log_files;
    }

    uid_t uid = getuid();
    while((ptr = readdir(pDir))!=0)
    {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
        {
            //std::cout << "file:" << ptr->d_name << std::endl;
            std::string tmp_file(ptr->d_name);

            //filename_t basename, ext;
            //std::tie(basename, ext) = details::file_helper::split_by_extension(base_filename_);

            //std::cout << "file_base_name_:" << file_base_name_ << std::endl;
            struct stat file_stat;
            const std::string &file_full_path = file_path_ + tmp_file; 
            if (stat(file_full_path.c_str(), &file_stat) == -1) {
                continue;
            }
            if (uid != file_stat.st_uid) {
                continue;
            }

            std::string head_old_type = file_base_name_ + "_log" ;
            std::size_t pos_old_type = tmp_file.find(head_old_type);
            if (pos_old_type != std::string::npos) // The log file of  old type, delete it.
            {
                (void)details::os::remove(file_path_ + tmp_file);
                continue;
            }

            std::string app_name = extractAppFromFilename(tmp_file);
            if(app_name.empty())
            {
                // 该文件格式不符合要求，pass
                // std::cout << "app_name is empty!" << std::endl;
                continue;
            }
            else
            {
                if (app_name != file_base_name_)
                {
                    // 该文件格式符合要求，APP_NAME和当前APP不匹配，pass
                    // std::cout << "app_name is : " << app_name << ", current app name is : " << file_base_name_ << std::endl;
                    continue;
                }
                else
                {
                    // 该文件格式符合要求，APP_NAME和当前APP匹配，返回指定下标
                    // std::cout << " app_name && current app name is same! " << std::endl;

                    std::string head = file_base_name_ + "_";
                    std::size_t pos_index = tmp_file.find(head);

                    if (pos_index != std::string::npos)
                    {
                        std::string strIndex = tmp_file.substr(pos_index + head.length(), 4);

                        std::uint32_t i = 0;
                        for (; i < strIndex.size(); i++)
                        {
                            if ((strIndex[i] < '0') || (strIndex[i] > '9'))
                            {
                                break;
                            }
                        }

                        /*Index data*/
                        if (i >= 4)
                        {
                            std::uint32_t file_index = std::stoi(strIndex, 0, 10);
                            //std::cout << "file_index:" << file_index << std::endl;
                            log_files.insert(std::pair<std::uint32_t, std::string> (file_index, tmp_file));
                        }

                    }
                }
            }
    	}
    }


    closedir(pDir);
    return log_files;
}


template<typename Mutex>
SPDLOG_INLINE filename_t hz_rotating_file_sink<Mutex>::current_writting_file_(std::map<std::uint32_t, std::string>& log_files)
{
    filename_t retFile = "";

    for (auto it = log_files.begin(); it != log_files.end(); it++)
    {
        filename_t basename, ext;
        std::tie(basename, ext) = details::file_helper::split_by_extension(it->second);

        //std::cout << "current_writting_file_: file:" << it->second << ", ext:" << ext << std::endl;

        if (ext == ".log")
        {
            //std::cout << "ext is .log" << std::endl;

            retFile = it->second;
            break;
        }
    }

    //std::cout << "current_writting_file_: retFile:" << retFile << std::endl;

    return retFile;
}

template<typename Mutex>
SPDLOG_INLINE void hz_rotating_file_sink<Mutex>::remove_old_files_(std::map<std::uint32_t, std::string>& log_files)
{
    int removecnt = log_files.size() - max_files_;
    if (removecnt <= 0) {
        return;
    }

    int orderbefore = 0;
    int orderafter = 0;
    int removebefore = 0;
    int removeafter = 0;
    int curindex = std::stoi(current_writting_filename_.data() + std::string(file_base_name_ + "_").size(), 0, 10);
    for (auto it : log_files) {
        it.first <= curindex ? ++orderbefore : ++orderafter;
    }
    // 汇总.log前后需要删除的文件个数
    if (orderafter > removecnt ) {
        removeafter = removecnt;
        removebefore = 0;
    }
    else {
        removeafter = orderafter;
        removebefore = removecnt - orderafter;
    }

    for (auto it = log_files.begin(); it != log_files.end();) {
        if (it->first <= curindex && removebefore > 0) {
            (void)details::os::remove(file_path_ + it->second);
            it = log_files.erase(it);
            --removebefore;
        }
        else if (it->first > curindex && removeafter > 0) {
            (void)details::os::remove(file_path_ + it->second);
            it = log_files.erase(it);
            --removeafter;
        }
        else {
            ++it;
        }
    }
}

template<typename Mutex>
SPDLOG_INLINE filename_t hz_rotating_file_sink<Mutex>::count_file_name_()
{
    filename_t retFile = "";
    std::map<std::uint32_t, std::string> log_files = current_files_();

    retFile = current_writting_file_(log_files);
    if (retFile != "")
    {
        /* Continue to write that file.*/
        //std::cout << "count_file_name_ 1 file:" << retFile << std::endl;

        current_writting_filename_ = retFile;
        return retFile;
    }
    else
    {
        time_t curtime;
        time(&curtime);
        tm* nowtime = localtime(&curtime);

        // filename_t basename, ext;
        // std::tie(basename, ext) = details::file_helper::split_by_extension(base_filename_);

        filename_t time_now = fmt_lib::format(SPDLOG_FILENAME_T("{0:0>4}-{1:0>2}-{2:0>2}_{3:0>2}-{4:0>2}-{5:0>2}"), 1900 + nowtime->tm_year, 1 + nowtime->tm_mon, nowtime->tm_mday, nowtime->tm_hour, nowtime->tm_min, nowtime->tm_sec);

        if (log_files.empty())
        {
            retFile = fmt_lib::format(SPDLOG_FILENAME_T("{0}_{1:0>4}_{2}{3}"), file_base_name_, 1, time_now, ".log");
            //std::cout << "count_file_name_ 2 file:" << retFile << std::endl;
        }
        else
        {
            std::uint32_t maxIndex = 0;

            auto iter_end = log_files.end();
            iter_end--;
            maxIndex = iter_end->first;

            if ((maxIndex < MAX_LOG_INDEX))
            {
                retFile = fmt_lib::format(SPDLOG_FILENAME_T("{0}_{1:0>4}_{2}{3}"), file_base_name_, maxIndex + 1, time_now, ".log");
                //std::cout << "count_file_name_ 3 file:" << retFile << std::endl;
            }
            else
            {
                std::uint32_t targetAddIndex = 1;
                for (; targetAddIndex < MAX_LOG_INDEX; ++targetAddIndex)
                {
                    auto iter = log_files.find(targetAddIndex);
                    if(iter == log_files.end())
                    {
                        break;  // found the target add index.
                    }
                }

                retFile = fmt_lib::format(SPDLOG_FILENAME_T("{0}_{1:0>4}_{2}{3}"), file_base_name_, targetAddIndex, time_now, ".log");
                //std::cout << "count_file_name_ 4 file:" << retFile << std::endl;
            }

        }
    }

    current_writting_filename_ = retFile;
    return retFile;
}

// delete the target if exists, and rename the src file  to target
// return true on success, false otherwise.
template<typename Mutex>
SPDLOG_INLINE bool hz_rotating_file_sink<Mutex>::rename_file_(const filename_t &src_filename, const filename_t &target_filename)
{
    // try to delete the target file in case it already exists.
    (void)details::os::remove(target_filename);
    return details::os::rename(src_filename, target_filename) == 0;
}

} // namespace sinks
} // namespace spdlog
