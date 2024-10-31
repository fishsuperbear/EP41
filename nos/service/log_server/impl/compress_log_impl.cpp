#include <zipper.h>

#include "log_server/impl/compress_log_impl.h"
#include "log_server/common/common_operation.h"
#include "log_server/log/log_server_logger.h"
#include "log_server/handler/log_server_fault_handler.h"
#include "log/src/spdlog/include/spdlog/details/os.h"


namespace hozon {
namespace netaos {
namespace logserver {

CompressLogImpl::CompressLogImpl()
:hozon::netaos::zmqipc::ZmqIpcServer()
, stopFlag_(false)
, request_queue_()
, request_queue_mutex_()
, process_threads_pool_()
, process_condition_()
{
}

int32_t
CompressLogImpl::Init()
{
    LOG_SERVER_INFO << "CompressLogImpl::Init";
    auto res = Start(compress_log_service_name);
    stopFlag_ = false;

    StartProcessThread();
    LOG_SERVER_INFO << "CompressLogImpl::Init Done";
    return res;
}

int32_t
CompressLogImpl::DeInit()
{
    LOG_SERVER_INFO << "CompressLogImpl::DeInit begin";
    stopFlag_ = true;

    process_condition_.notify_all();
    for (auto& thread : process_threads_pool_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    LOG_SERVER_DEBUG << "ZMQ server stop begin";
    auto res = Stop();
    LOG_SERVER_DEBUG << "ZMQ server stop end";

    LOG_SERVER_INFO << "CompressLogImpl::DeInit end";
    return res;
}


int32_t
CompressLogImpl::Process(const std::string& request, std::string& reply)
{
    LOG_SERVER_DEBUG << "CompressLogImpl::Process";

    // 将新的请求添加到队列中
    std::unique_lock<std::mutex> lock(request_queue_mutex_);
    request_queue_.push_back(request);
    process_condition_.notify_one();
    lock.unlock();

    reply.clear();
    return 0;
}

void
CompressLogImpl::CompressFile(const std::string& request)
{
    ZipperInfo info;
    info.ParseFromString(request);
    std::string filePath = info.file_path();
    std::string file_base_name = info.file_base_name();
    std::string filename = info.filename();
    std::string basename = info.basename();

    LOG_SERVER_INFO << "ParseFromArray() filePath is : " << filePath;
    LOG_SERVER_INFO << "ParseFromArray() file_base_name is : " << file_base_name;
    LOG_SERVER_INFO << "ParseFromArray() filename is : " << filename;
    LOG_SERVER_INFO << "ParseFromArray() basename is : " << basename;

    zipper::Zipper zipfile(filePath + "tmp_" + file_base_name + ".zip");
    zipfile.add(filename);
    zipfile.close();

    rename_file_(filePath + "tmp_" + file_base_name + ".zip", filePath + basename + ".zip");
    PathRemove(filename);
}

bool
CompressLogImpl::rename_file_(const spdlog::filename_t &src_filename, const spdlog::filename_t &target_filename)
{
    (void)spdlog::details::os::remove(target_filename);
    return spdlog::details::os::rename(src_filename, target_filename) == 0;
}

void CompressLogImpl::StartProcessThread()
{
    const int threads_num = 5;
    for (int i = 0; i < threads_num; ++i) {
        process_threads_pool_.push_back(std::thread([&]() {
            while (!stopFlag_) {
                std::unique_lock<std::mutex> lock(request_queue_mutex_);
                process_condition_.wait(lock, [&]() {
                    return (!request_queue_.empty() || stopFlag_);
                }); // 等待新的请求

                if (stopFlag_) {
                    break;
                }

                std::string request = request_queue_.front(); // 获取队首请求
                request_queue_.pop_front(); // 出队
                lock.unlock(); // 释放锁

                // 处理请求
                CompressFile(request);
            }
        }));
    }
}


}  // namespace logserver
}  // namespace netaos
}  // namespace hozon
