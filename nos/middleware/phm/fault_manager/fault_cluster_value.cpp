/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: fault event inhibit
*/
#include "phm/common/include/phm_logger.h"
#include "phm/common/include/phm_config.h"
#include "phm/fault_manager/include/fault_cluster_value.h"
#include <bitset>

namespace hozon {
namespace netaos {
namespace phm {

static const int CLUSTER_TABLE_SIZE = 60U;
static const int CLUSTER_VALUE_DEFAULT = 0x00;

struct FaultBitPositionInfo
{
    uint8_t faultStatus;
    uint32_t faultBitPosition;
};

struct Notify2App
{
    std::string clusterName;
    uint32_t clusterValue;
    std::unordered_map<uint32_t, FaultBitPositionInfo> faultMap;
};

Notify2App ClusterTable[CLUSTER_TABLE_SIZE] =
{
    { "platform", CLUSTER_VALUE_DEFAULT },
    { "30-camera", CLUSTER_VALUE_DEFAULT },
    { "120-camera", CLUSTER_VALUE_DEFAULT },
    { "rf-camera", CLUSTER_VALUE_DEFAULT },
    { "lr-camera", CLUSTER_VALUE_DEFAULT },
    { "rr-camera", CLUSTER_VALUE_DEFAULT },
    { "lf-camera", CLUSTER_VALUE_DEFAULT },
    { "r-camera", CLUSTER_VALUE_DEFAULT },
    { "lidar", CLUSTER_VALUE_DEFAULT },
    { "f-radar", CLUSTER_VALUE_DEFAULT },
    { "rf-radar", CLUSTER_VALUE_DEFAULT },
    { "lf-radar", CLUSTER_VALUE_DEFAULT },
    { "rr-radar", CLUSTER_VALUE_DEFAULT },
    { "lr-radar", CLUSTER_VALUE_DEFAULT },
    { "gnss", CLUSTER_VALUE_DEFAULT },
    { "imu", CLUSTER_VALUE_DEFAULT },
    { "chassis", CLUSTER_VALUE_DEFAULT },
    { "hm-map", CLUSTER_VALUE_DEFAULT },
    { "hq-map", CLUSTER_VALUE_DEFAULT },
    { "dr", CLUSTER_VALUE_DEFAULT },
    { "local-location", CLUSTER_VALUE_DEFAULT },
    { "global-location", CLUSTER_VALUE_DEFAULT },
    { "fusion-map", CLUSTER_VALUE_DEFAULT },
    { "local-map", CLUSTER_VALUE_DEFAULT },
    { "flm", CLUSTER_VALUE_DEFAULT },
    { "flc", CLUSTER_VALUE_DEFAULT },
    { "frc", CLUSTER_VALUE_DEFAULT },
    { "frm", CLUSTER_VALUE_DEFAULT },
    { "rlm", CLUSTER_VALUE_DEFAULT },
    { "rlc", CLUSTER_VALUE_DEFAULT },
    { "rrc", CLUSTER_VALUE_DEFAULT },
    { "rrm", CLUSTER_VALUE_DEFAULT },
    { "sfr", CLUSTER_VALUE_DEFAULT },
    { "sfl", CLUSTER_VALUE_DEFAULT },
    { "srr", CLUSTER_VALUE_DEFAULT },
    { "srl", CLUSTER_VALUE_DEFAULT },
    { "front-fisheye-camera", CLUSTER_VALUE_DEFAULT },
    { "left-fisheye-camera", CLUSTER_VALUE_DEFAULT },
    { "right-fisheye-camera", CLUSTER_VALUE_DEFAULT },
    { "rear-fisheye-camera", CLUSTER_VALUE_DEFAULT },
    { "fusion-obj", CLUSTER_VALUE_DEFAULT },
    { "fusion-parkinglot", CLUSTER_VALUE_DEFAULT },
    { "uss-parkinglot", CLUSTER_VALUE_DEFAULT },
    { "uss-obstacle", CLUSTER_VALUE_DEFAULT },
    { "freespace", CLUSTER_VALUE_DEFAULT },
    { "hpp-location", CLUSTER_VALUE_DEFAULT },
    { "slam", CLUSTER_VALUE_DEFAULT },

    { "reserved1", CLUSTER_VALUE_DEFAULT },
    { "reserved2", CLUSTER_VALUE_DEFAULT },
    { "reserved3", CLUSTER_VALUE_DEFAULT },
    { "reserved4", CLUSTER_VALUE_DEFAULT },
    { "reserved5", CLUSTER_VALUE_DEFAULT },
    { "reserved6", CLUSTER_VALUE_DEFAULT },
    { "reserved7", CLUSTER_VALUE_DEFAULT },
    { "reserved8", CLUSTER_VALUE_DEFAULT },
    { "reserved9", CLUSTER_VALUE_DEFAULT },
    { "reserved10", CLUSTER_VALUE_DEFAULT },
    { "reserved11", CLUSTER_VALUE_DEFAULT },
    { "reserved12", CLUSTER_VALUE_DEFAULT },
    { "reserved13", CLUSTER_VALUE_DEFAULT }
};


void
FaultClusterValue::UpdateClusterValue(const int tableIndex)
{
    // default
    ClusterTable[tableIndex].clusterValue = CLUSTER_VALUE_DEFAULT;

    // update all fault bitposition
    for (auto& fault : ClusterTable[tableIndex].faultMap) {
        // set bitPosition 1
        std::bitset<8> clusterValueBitSet(ClusterTable[tableIndex].clusterValue);
        clusterValueBitSet[fault.second.faultBitPosition] = 1;
        ClusterTable[tableIndex].clusterValue = static_cast<uint8_t>(clusterValueBitSet.to_ulong());
    }

    return;
}

void
FaultClusterValue::UpdateTableItemFault(const int tableIndex, const uint32_t faultKey, const uint8_t faultStatus, int bitPosition)
{
    // update table fault
    auto iter = ClusterTable[tableIndex].faultMap.find(faultKey);
    if (iter == ClusterTable[tableIndex].faultMap.end()) {
        if (1 == faultStatus) {
            FaultBitPositionInfo cFaultLevelInfo;
            cFaultLevelInfo.faultStatus = faultStatus;
            cFaultLevelInfo.faultBitPosition = bitPosition;
            ClusterTable[tableIndex].faultMap.insert(std::make_pair(faultKey, cFaultLevelInfo));
        }
        else {
            // do nothing
        }
    }
    else {
        if (iter->second.faultStatus != faultStatus) {
            ClusterTable[tableIndex].faultMap.erase(faultKey);
        }
    }

    return;
}

void
FaultClusterValue::UpdateClusterData(const uint32_t faultKey, const uint8_t faultStatus, std::vector<FaultClusterItem>& FaultCluster)
{
    PHM_INFO << "FaultClusterValue::UpdateClusterData faultKey " << faultKey << " faultStatus " << faultStatus;
    for (auto& clusterItem : FaultCluster) {
        int index = 0;
        for (; index < CLUSTER_TABLE_SIZE; ++index) {
            if (clusterItem.clusterName == ClusterTable[index].clusterName) {
                break;
            }
        }

        if (CLUSTER_TABLE_SIZE == index) {
            continue;
        }

        // add or delete fault from faultMap
        UpdateTableItemFault(index, faultKey, faultStatus, clusterItem.bitPosition);

        // update cluster value
        UpdateClusterValue(index);
    }

    return;
}

void
FaultClusterValue::GetClusterValueData(std::vector<ClusterItem>& clusterItem)
{
    // get all cur level
    std::string allClusterValueInfo;
    for (size_t i = 0; i < CLUSTER_TABLE_SIZE; ++i) {
        ClusterItem tempClusterItem;
        tempClusterItem.clusterName = ClusterTable[i].clusterName;
        tempClusterItem.clusterValue = static_cast<uint8_t>(ClusterTable[i].clusterValue);
        allClusterValueInfo += " " + std::to_string(ClusterTable[i].clusterValue);
        clusterItem.emplace_back(tempClusterItem);
    }

    PHM_INFO << "FaultClusterValue::GetClusterValueData allClusterValueInfo:" << allClusterValueInfo;
    return;
}

void
FaultClusterValue::GetClusterValueData(std::string clusterName, ClusterItem& clusterItem)
{
    for (size_t i = 0; i < CLUSTER_TABLE_SIZE; ++i) {
        if (clusterName == ClusterTable[i].clusterName) {
            clusterItem.clusterName = ClusterTable[i].clusterName;
            clusterItem.clusterValue = static_cast<uint8_t>(ClusterTable[i].clusterValue);
            break;
        }
    }

    return;
}

}  // namespace phm
}  // namespace netaos
}  // namespace hozon