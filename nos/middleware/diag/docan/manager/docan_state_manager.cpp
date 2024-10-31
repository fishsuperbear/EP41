/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanStateManager implement
 */

#include "docan_state_manager.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanStateManager* DocanStateManager::s_instance = nullptr;
    std::mutex DocanStateManager::s_instance_mutex;

    DocanStateManager* DocanStateManager::instance()
    {
        std::lock_guard<std::mutex> sync(s_instance_mutex);
        if (nullptr == s_instance) {
            s_instance = new DocanStateManager;
        }
        return s_instance;
    }

    void DocanStateManager::destroy()
    {
        std::lock_guard<std::mutex> sync(s_instance_mutex);
        if (nullptr != s_instance) {
            delete s_instance;
            s_instance = nullptr;
        }
    }

    DocanStateManager::DocanStateManager()
    {
    }

    DocanStateManager::~DocanStateManager()
    {
    }

    int32_t DocanStateManager::Init(void)
    {
        return 0;
    }

    int32_t DocanStateManager::Start(void)
    {
        return 0;
    }

    int32_t DocanStateManager::Stop(void)
    {
        return 0;
    }

    int32_t DocanStateManager::Deinit(void)
    {
        return 0;
    }




} // end of diag
} // end of netaos
} // end of hozon
/* EOF */
