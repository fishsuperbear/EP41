#include "adf/include/node_skeleton.h"
#include "adf/include/internal_log.h"
#include "idl/generated/common.h"

namespace hozon {
namespace netaos {
namespace adf {

NodeSkeletonBase::NodeSkeletonBase(const NodeConfig::CommInstanceConfig& config)
    : _container(0, config.buffer_capacity),
      _need_stop(false),
      _is_async(config.is_async),
      _config(config),
      _freq_monitor(std::string("(S)") + config.name, config.exp_freq, config.freq_threshold_ratio) {
    if (config.freq_valid) {
        _freq_monitor.EnablePeriodCheck();
    }
    _freq_monitor.Start();
    MonitorReader::GetInstance().RegisterMonitor(&_freq_monitor);

    if (_is_async) {
        _thread = std::make_shared<std::thread>(&NodeSkeletonBase::DataSendRoutine, this);
    }
}

NodeSkeletonBase::~NodeSkeletonBase() {}

int32_t NodeSkeletonBase::SendOneData(BaseDataTypePtr data) {
    if (data == nullptr) {
        ADF_LOG_ERROR << "The data ptr for sending is nullptr";
        return -1;
    } else if (_need_stop) {
        return -1;
    }

    _freq_monitor.PushOnce();

    if (_is_async) {
        std::lock_guard<std::mutex> lk(_send_mtx);
        _container.Push(data);
        _send_cv.notify_one();
    } else {
        _container.Push(data);
        OnDataNeedSend();
        // clear 旧数据，以及超过容量限制的数据
    }
    return 0;
}

void NodeSkeletonBase::DataSendRoutine() {
    while (!_need_stop) {
        std::unique_lock<std::mutex> lck(_send_mtx);
        while (_container.Empty()) {
            _send_cv.wait(lck);
            if (_need_stop) {
                return;
            }
        }
        OnDataNeedSend();
    }
}

void NodeSkeletonBase::Stop() {
    _freq_monitor.Stop();

    _need_stop = true;
    if (_is_async) {
        _send_cv.notify_all();
        if (_thread->joinable()) {
            _thread->join();
        }
    }
    Deinit();
}
}  //  namespace adf
}  //  namespace netaos
}  //  namespace hozon
