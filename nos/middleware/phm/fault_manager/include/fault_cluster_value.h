
#pragma once
#include "phm/include/phm_def.h"
#include "phm/common/include/phm_common_def.h"
#include <vector>
#include <string>

namespace hozon {
namespace netaos {
namespace phm {


class FaultClusterValue
{
public:
    FaultClusterValue() {}
    ~FaultClusterValue() {}

    void UpdateClusterValue(const int tableIndex);
    void UpdateTableItemFault(const int tableIndex, const uint32_t faultKey, const uint8_t faultStatus, int bitPosition);
    void UpdateClusterData(const uint32_t faultKey, const uint8_t faultStatus, std::vector<FaultClusterItem>& FaultCluster);
    void GetClusterValueData(std::vector<ClusterItem>& clusterItem);
    void GetClusterValueData(std::string clusterName, ClusterItem& clusterItem);
};


}  // namespace phm
}  // namespace netaos
}  // namespace hozon

