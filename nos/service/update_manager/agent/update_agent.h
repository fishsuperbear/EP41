/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: UA module
 */
#ifndef UPDATE_AGENT_H
#define UPDATE_AGENT_H

#include <stdint.h>

#include "diag/libsttask/STTaskRunner.h"
#include "update_manager/common/data_def.h"

using namespace hozon::netaos::sttask;

namespace hozon {
namespace netaos {
namespace update {

enum class UpdateSuccessFlag : uint16_t {
    LIDAR_FL_SUCCESS        = 0x0001,
    LIDAR_FR_SUCCESS        = 0x0002,
    SRR_FL_SUCCESS          = 0x0004,
    SRR_FR_SUCCESS          = 0x0008,
    SRR_RL_SUCCESS          = 0x0010,
    SRR_RR_SUCCESS          = 0x0020,
    USSC_SUCCESS            = 0x0040,
    MDC_UPDATE_SUCCESS      = 0x0080,
    MDC_ACTIVE_SUCCESS      = 0x0100,
};

class UpdateAgent : public STTaskRunner {
public:
    static UpdateAgent &Instance()
    {
        static UpdateAgent instance;
        return instance;
    }

    int32_t Init();
    int32_t Start();
    int32_t Stop();
    int32_t Deinit();

    uint8_t Update();
    uint8_t Active();
    uint8_t RollBack();
    uint8_t Cancel();

    virtual const char* getTaskThreadName() { return "ua_thread"; };

protected:
        virtual void        onInit();
        virtual uint32_t    onOperationStart(uint32_t operationId, STNormalTask* topTask);
        virtual void        onOperationEnd(uint32_t operationId, uint32_t result, STNormalTask* topTask);
        virtual uint32_t    onStepStart(uint32_t operationId, uint32_t stepId, STStepTask* stepTask);
        virtual void        onStepEnd(uint32_t operationId, uint32_t stepId, uint32_t result, STStepTask* stepTask);
        virtual void        onUnexpectedEvent(uint32_t eventKind, uint32_t eventId, STEvent* event);

private:
    UpdateAgent();
    UpdateAgent(const UpdateAgent &);
    UpdateAgent & operator = (const UpdateAgent &);

    uint16_t update_success_flag_;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // UPDATE_AGENT_H
