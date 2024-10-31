#pragma once

#include "adf-lite/include/executor.h"
#include "adf-lite/ds/ds_config.h"
#include "adf-lite/ds/ds_recv/ds_recv.h"
#include "adf-lite/ds/ds_send/ds_send.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class DataSourceExecutor : public Executor {
public:
    DataSourceExecutor();
    ~DataSourceExecutor();

    int32_t AlgInit();
    void AlgPreRelease();
    void AlgRelease();

private:
    int32_t ParseConfig(const std::string& file);

    DSConfig _ds_config;
    std::unordered_map<std::string, std::shared_ptr<DsRecv>> _recvs;
    std::unordered_map<std::string, std::shared_ptr<DsSend>> _sends;
};

REGISTER_ADF_CLASS(DataSource, DataSourceExecutor)

}
}
}

