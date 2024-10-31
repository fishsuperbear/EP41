// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)

#pragma once

#include <spdlog/sinks/base_sink.h>
#include <spdlog/details/file_helper.h>
#include <spdlog/details/null_mutex.h>
#include <spdlog/details/synchronous_factory.h>

#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <map>
#ifdef BUILD_LOG_SERVER_ENABLE
    #include "zmq_ipc/manager/zmq_ipc_client.h"
#endif

namespace spdlog {
namespace sinks {

//
// Rotating file sink based on size
//
template<typename Mutex>
class hz_rotating_file_sink final : public base_sink<Mutex>
{
public:
    hz_rotating_file_sink(filename_t base_file_path, filename_t file_base_name, std::size_t max_size, std::size_t max_files, bool rotate_on_open = false,
        const file_event_handlers &event_handlers = {});
    ~hz_rotating_file_sink();
    filename_t filename();

protected:
    void sink_it_(const details::log_msg &msg) override;
    void flush_() override;
    

private:
    void rotate_();
    filename_t count_file_name_();
    filename_t current_writting_file_(std::map<std::uint32_t, std::string>& log_files);
    void remove_old_files_(std::map<std::uint32_t, std::string>& log_files);
    std::map<std::uint32_t, std::string> current_files_();

    void compress_file_(const filename_t &filename, const filename_t &basename);
    std::string GetAbsolutePath(const std::string& relativePath);
    void log_file_create_();

    // delete the target if exists, and rename the src file  to target
    // return true on success, false otherwise.
    bool rename_file_(const filename_t &src_filename, const filename_t &target_filename);
    std::string extractAppFromFilename(const std::string& filename);

    filename_t file_base_name_;
    filename_t file_path_;
    filename_t current_writting_filename_;
    std::size_t max_size_;
    std::size_t max_files_;
    std::size_t current_size_;
    details::file_helper file_helper_;
    bool rotate_on_open_;
#ifdef BUILD_LOG_SERVER_ENABLE
    std::unique_ptr<hozon::netaos::zmqipc::ZmqIpcClient> client_;
#endif
};

using hz_rotating_file_sink_mt = hz_rotating_file_sink<std::mutex>;
using hz_rotating_file_sink_st = hz_rotating_file_sink<details::null_mutex>;

} // namespace sinks

//
// factory functions
//

template<typename Factory = spdlog::synchronous_factory>
inline std::shared_ptr<logger> rotating_logger_mt(const std::string &logger_name, const filename_t &filename, size_t max_file_size,
    size_t max_files, bool rotate_on_open = false, const file_event_handlers &event_handlers = {})
{
    return Factory::template create<sinks::hz_rotating_file_sink_mt>(
        logger_name, filename, max_file_size, max_files, rotate_on_open, event_handlers);
}

template<typename Factory = spdlog::synchronous_factory>
inline std::shared_ptr<logger> rotating_logger_st(const std::string &logger_name, const filename_t &filename, size_t max_file_size,
    size_t max_files, bool rotate_on_open = false, const file_event_handlers &event_handlers = {})
{
    return Factory::template create<sinks::hz_rotating_file_sink_st>(
        logger_name, filename, max_file_size, max_files, rotate_on_open, event_handlers);
}
} // namespace spdlog

#ifdef SPDLOG_HEADER_ONLY
#    include "hz_rotating_file_sink-inl.h"
#endif
