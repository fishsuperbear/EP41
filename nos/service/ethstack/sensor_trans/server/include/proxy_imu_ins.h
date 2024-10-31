#pragma once

#include "ara/com/sample_ptr.h"
#include "proto/soc/sensor_imu_ins.pb.h"
#include "hozon/netaos/v1/mcudataservice_proxy.h"

namespace hozon {
namespace netaos {
namespace sensor {

class ImuInsProxy {
public:
    std::shared_ptr<hozon::soc::ImuIns> Trans(
            ara::com::SamplePtr<::hozon::netaos::AlgImuInsInfo const> imuIns_info);
    ImuInsProxy();
    ~ImuInsProxy() = default;
private:
    // int Init() override;
    // void Deinit() override;
    void PrintOriginalData(ara::com::SamplePtr<hozon::netaos::AlgImuInsInfo const> imuIns_info);
    uint32_t imu_seqid;
    uint64_t imu_ins_pub_last_time;
    uint32_t _imuins_pub_last_seq;
};

}
}
}


