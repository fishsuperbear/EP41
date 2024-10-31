#pragma once

#include <memory>
#include "proto/soc/sensor_gnss.pb.h"
#include "hozon/netaos/v1/mcudataservice_proxy.h"

namespace hozon {
namespace netaos {
namespace sensor {

class ProxyGnss {
public:
    ProxyGnss();
    ~ProxyGnss() = default;
    std::shared_ptr<hozon::soc::gnss::GnssInfo> Trans(
        ara::com::SamplePtr<::hozon::netaos::AlgGnssInfo const> data);
private:
    uint32_t _gnss_pub_last_seq;
    
};

}   // namespace sensor
}   // namespace netaos
}   // namespace hozon