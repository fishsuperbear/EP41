
#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include "adf/include/data_source_monitor.h"
#include "adf/include/data_types/common/types.h"
#include "adf/include/node_config.h"
#include "adf/include/thread_safe_stack.h"
#include "cm/include/skeleton.h"
#include "fastdds/dds/topic/TopicDataType.hpp"
#include "idl/generated/common.h"

namespace hozon {
namespace netaos {
namespace adf {

class NodeSkeletonBase {
   public:
    explicit NodeSkeletonBase(const NodeConfig::CommInstanceConfig& config);
    ~NodeSkeletonBase();

    void DataSendRoutine();
    int32_t SendOneData(BaseDataTypePtr data);
    void Stop();

   protected:
    virtual void OnDataNeedSend() = 0;
    virtual void Deinit() = 0;

    uint32_t _buffer_depth;
    hozon::netaos::adf::ThreadSafeStack<BaseDataTypePtr> _container;
    std::mutex _send_mtx;
    std::condition_variable _send_cv;
    std::shared_ptr<std::thread> _thread;
    std::atomic<bool> _need_stop;
    static bool _mbuf_inited;
    bool _is_async;
    NodeConfig::CommInstanceConfig _config;
    FreqMonitor _freq_monitor;
};
}  // namespace adf
}  // namespace netaos
}  // namespace hozon
