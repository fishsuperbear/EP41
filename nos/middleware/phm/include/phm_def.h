/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: phm def
 */

#ifndef PHM_DEF_H
#define PHM_DEF_H

#include <stdint.h>
#include <vector>
#include <string>

namespace hozon {
namespace netaos {
namespace phm {

enum class DebounceType : uint8_t {
    DEBOUNCE_TYPE_UNUSE = 0,
    DEBOUNCE_TYPE_COUNT,
    DEBOUNCE_TYPE_TIME
};

struct CountDebounce {
    uint32_t debounceTime;
    uint32_t debounceCount;
};

struct TimeDebounce {
    uint32_t debounceTime;
};

typedef union FaultDebounceSetting {
    CountDebounce countDebounce;
    TimeDebounce timeDebounce;
} FaultDebounceSetting_t;

typedef struct FaultDebounce {
    DebounceType debounceType;
    FaultDebounceSetting_t debounceSetting;
} FaultDebounce_t;

typedef struct SendFault {
    uint32_t faultId;
    uint8_t faultObj;
    uint8_t faultStatus;
    FaultDebounce_t faultDebounce;
    bool isAutoRecovery = false;
    uint32_t autoRecoveryTime = 0;      // ms

    SendFault(uint32_t _faultId, uint8_t _faultObj, uint8_t _faultStatus, bool _isAutoRecovery = false, uint32_t _autoRecoveryTime = 0) {
        faultId = _faultId;
        faultObj = _faultObj;
        faultStatus = _faultStatus;
        faultDebounce.debounceType = DebounceType::DEBOUNCE_TYPE_UNUSE;
        faultDebounce.debounceSetting = {0x00};
        isAutoRecovery = _isAutoRecovery;
        autoRecoveryTime = _autoRecoveryTime;
    }
} SendFault_t;

struct ClusterItem {
    std::string clusterName;
    uint8_t clusterValue;
};

typedef struct ReceiveFault {
    std::string faultDomain;
    uint64_t faultOccurTime;
    uint32_t faultId;
    uint8_t faultObj;
    uint8_t faultStatus;
    uint32_t faultCombinationId;
    std::string faultDes;
    std::vector<ClusterItem> faultCluster;
} ReceiveFault_t;

struct phm_transition {
    uint32_t checkpointSrcId;
    uint32_t checkpointDestId;

    bool operator == (const phm_transition& t) const
    {
        return (checkpointSrcId == t.checkpointSrcId) && (checkpointDestId == t.checkpointDestId);
    }

    bool operator < (const phm_transition& t) const
    {
        if (checkpointSrcId < t.checkpointSrcId)
        {
            return true;
        }
        else if (t.checkpointSrcId < checkpointSrcId)
        {
            return false;
        }
        else
        {
            return checkpointDestId < t.checkpointDestId;
        }
    }
};

}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_DEF_H
