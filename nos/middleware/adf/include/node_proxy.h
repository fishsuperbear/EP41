#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>

#include "adf/include/data_source_monitor.h"
#include "adf/include/data_types/common/types.h"
#include "adf/include/node_config.h"
#include "adf/include/thread_safe_stack.h"
#include "idl/generated/common.h"

namespace hozon {
namespace netaos {
namespace adf {

class NodeProxyBase {
   public:
    explicit NodeProxyBase(const NodeConfig::CommInstanceConfig& config);
    virtual ~NodeProxyBase();

    BaseDataTypePtr GetOneDataBlocking(const uint32_t blockTimeout = UINT32_MAX);
    BaseDataTypePtr GetOneData(const uint32_t freshDataTime = UINT32_MAX);
    BaseDataTypePtr PopOneData();
    std::vector<BaseDataTypePtr> GetNdata(const size_t n);
    std::vector<BaseDataTypePtr> PopNdata(const size_t n);

    virtual void PauseReceive() = 0;
    virtual void ResumeReceive() = 0;
    virtual void Deinit() = 0;

    void Stop();

   protected:
    virtual void OnDataReceive() = 0;

    void PushOneAndNotify(BaseDataTypePtr data_ptr);
    void ClearContainerData();

    uint32_t _buffer_depth;
    ThreadSafeStack<BaseDataTypePtr> _container;
    std::mutex _recv_mtx;
    std::condition_variable _recv_cv;
    bool _stop_flag = false;
    static bool _is_rm_inited;
    bool _new_data_arrived = false;
    bool _is_async;
    FreqMonitor _freq_monitor;
    NodeConfig::CommInstanceConfig _config;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
