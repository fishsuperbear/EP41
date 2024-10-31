#include "adf-lite/ds/ds_executor.h"
#include "adf-lite/ds/ds_recv/idl_cuda_ds_recv.h"
#include "adf-lite/ds/ds_recv/idl_ds_recv.h"
#include "adf-lite/ds/ds_recv/nvs_cuda_ds_recv.h"
#include "adf-lite/ds/ds_recv/proto_cm_ds_recv.h"
#include "adf-lite/ds/ds_recv/proto_cuda_ds_recv.h"
#include "adf-lite/ds/ds_send/proto_cm_ds_send.h"
#include "adf-lite/include/topic_manager.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

DataSourceExecutor::DataSourceExecutor() {}

DataSourceExecutor::~DataSourceExecutor() {}

template <typename T>
std::shared_ptr<DsRecv> CreateCMDsRecv(const DSConfig::DataSource& config) {
    return std::make_shared<T>(config);
}

std::unordered_map<std::string, std::function<std::shared_ptr<DsRecv>(const DSConfig::DataSource&)>> g_ds_recv_map = {
    {"proto_proxy", CreateCMDsRecv<ProtoCMDsRecv>},         {"proto_cuda_proxy", CreateCMDsRecv<ProtoCudaDsRecv>},
    {"idl_cuda_proxy", CreateCMDsRecv<IdlCudaDsRecv>},      {"idl_proxy", CreateCMDsRecv<IdlDsRecv>},
#ifdef BUILD_FOR_ORIN
    {"nvs_cuda_proxy", CreateCMDsRecv<NvsCudaDesayDsRecv>},
#endif
};

template <typename T>
std::shared_ptr<DsSend> CreateCMDsSend(const DSConfig::DataSource& config) {
    return std::make_shared<T>(config);
}

std::unordered_map<std::string, std::function<std::shared_ptr<DsSend>(const DSConfig::DataSource&)>> g_ds_send_map = {
    {"proto_skeleton", CreateCMDsSend<ProtoCMDsSend>},
};

int32_t DataSourceExecutor::AlgInit() {
    if (_ds_config.Parse(GetConfigFilePath()) < 0) {
        return -1;
    }
    DsLogger::GetInstance()._logger.Init("DS", static_cast<hozon::netaos::adf_lite::LogLevel>(_ds_config.log.level));

    for (auto& ds : _ds_config.data_sources_in) {
        if (g_ds_recv_map.find(ds.type) != g_ds_recv_map.end()) {
            DS_LOG_DEBUG << "Create CM data source of type " << ds.type << " for topic: " << ds.topic;
            _recvs[ds.topic] = g_ds_recv_map[ds.type](ds);
            TopicManager::GetInstance().AddRecvInstance(ds.topic, ds.cm_topic, _recvs[ds.topic]);
        } else {
            DS_LOG_ERROR << "Fail to find data source type " << ds.type;
            return -1;
        }
    }

    for (auto& ds : _ds_config.data_sources_out) {
        if (g_ds_send_map.find(ds.type) != g_ds_send_map.end()) {
            DS_LOG_DEBUG << "Create CM data source of type " << ds.type;
            _sends[ds.topic] = g_ds_send_map[ds.type](ds);
        } else {
            DS_LOG_ERROR << "Fail to find data source type " << ds.type;
            return -1;
        }
    }
    return 0;
}

void DataSourceExecutor::AlgPreRelease() {
    for (auto& ds : _ds_config.data_sources_out) {
        if (_sends[ds.topic]) {
            _sends[ds.topic]->PreDeinit();
        }
    }
}

void DataSourceExecutor::AlgRelease() {
    for (auto& ds : _ds_config.data_sources_in) {
        if (_recvs[ds.topic]) {
            _recvs[ds.topic]->Deinit();
        }
    }
    for (auto& ds : _ds_config.data_sources_out) {
        if (_sends[ds.topic]) {
            _sends[ds.topic]->Deinit();
        }
    }
}

}  // namespace adf_lite
}  // namespace netaos
}  // namespace hozon
