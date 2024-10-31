#include "adf/include/node_proxy.h"
#include "adf/include/internal_log.h"

namespace hozon {
namespace netaos {
namespace adf {

NodeProxyBase::NodeProxyBase(const NodeConfig::CommInstanceConfig& config)
    : _buffer_depth(config.buffer_capacity),
      _container(0, config.buffer_capacity),
      _is_async(config.is_async),
      _freq_monitor(std::string("(R)") + config.name, config.exp_freq, config.freq_threshold_ratio),
      _config(config) {
    if (config.freq_valid) {
        _freq_monitor.EnablePeriodCheck();
    }

    MonitorReader::GetInstance().RegisterMonitor(&_freq_monitor);
}

NodeProxyBase::~NodeProxyBase() {}

BaseDataTypePtr NodeProxyBase::GetOneDataBlocking(const uint32_t blockTimeout) {
    if (!_is_async) {
        // report fault
        ADF_LOG_CRITICAL << "Get blocking data is not supported in sync mode.";
        return nullptr;
    }

    std::unique_lock<std::mutex> recvLk(_recv_mtx);
    if (_stop_flag) {
        ADF_LOG_INFO << "Data receive got stop flag before wait.";
        return nullptr;
    }
    if (!_new_data_arrived) {
        if (_recv_cv.wait_for(recvLk, std::chrono::milliseconds(blockTimeout)) == std::cv_status::timeout) {
            ADF_LOG_DEBUG << "No new data arrived since this api called. timeout(ms): " << blockTimeout;
            return nullptr;
        }
    }

    _new_data_arrived = false;
    if (_stop_flag) {
        ADF_LOG_INFO << "Data receive got stop Flag";
        return nullptr;
    }
    auto dataPtr = _container.GetOneData();
    recvLk.unlock();
    if (dataPtr == nullptr) {
        return nullptr;
    }
    return *dataPtr;
}

BaseDataTypePtr NodeProxyBase::GetOneData(const uint32_t freshDataTime) {
    if (!_is_async) {
        OnDataReceive();
    }

    if (_stop_flag) {
        ADF_LOG_INFO << "Data receive got stopFlag signal";
        return nullptr;
    }
    auto dataPtr = _container.GetOneData(freshDataTime);
    if (dataPtr == nullptr) {
        return nullptr;
    }
    return *dataPtr;
}

BaseDataTypePtr NodeProxyBase::PopOneData() {
    if (!_is_async) {
        OnDataReceive();
    }

    if (_stop_flag) {
        ADF_LOG_INFO << "Data receive got stopFlag signal";
        return nullptr;
    }
    auto dataPtr = _container.PopFront();
    if (dataPtr == nullptr) {
        return nullptr;
    }
    return *dataPtr;
}

std::vector<BaseDataTypePtr> NodeProxyBase::GetNdata(const size_t n) {
    if (!_is_async) {
        OnDataReceive();
    }

    if (_stop_flag) {
        ADF_LOG_INFO << "Data receive got stopFlag";
        return {};
    }

    auto result = _container.GetNdata(n);

    std::vector<BaseDataTypePtr> output;
    output.reserve(result.size());

    for (auto& it : result) {
        if (it != nullptr) {
            output.push_back(*it);
        }
    }
    return output;
}

std::vector<BaseDataTypePtr> NodeProxyBase::PopNdata(const size_t n) {
    if (!_is_async) {
        OnDataReceive();
    }

    if (_stop_flag) {
        ADF_LOG_INFO << "Data receive got stopFlag";
        return {};
    }

    auto result = _container.PopNdata(n);

    std::vector<BaseDataTypePtr> output;
    output.reserve(result.size());

    for (auto& it : result) {
        if (it != nullptr) {
            output.push_back(*it);
        }
    }
    return output;
}

void NodeProxyBase::Stop() {
    {
        std::unique_lock<std::mutex> recvLk(_recv_mtx);
        _stop_flag = true;
        _recv_cv.notify_all();
    }
    Deinit();
}

void NodeProxyBase::PushOneAndNotify(BaseDataTypePtr data_ptr) {
    std::unique_lock<std::mutex> cv_lock(_recv_mtx);
    _container.Push(data_ptr);
    _new_data_arrived = true;
    _recv_cv.notify_all();
}

void NodeProxyBase::ClearContainerData() {
    std::unique_lock<std::mutex> cv_lock(_recv_mtx);
    _container.Clear();
    _new_data_arrived = false;
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
