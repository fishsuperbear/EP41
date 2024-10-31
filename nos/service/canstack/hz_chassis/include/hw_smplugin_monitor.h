/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Description: config server status monitor
 */

#pragma once


namespace hozon {
namespace chassis {


class SmPluginMonitor {

public:
	static SmPluginMonitor &Instance()
    {
        static SmPluginMonitor instance;
        return instance;
    }
    void Init();
    void DeInit();

    bool InitSmPlugin();
    bool RegisterProcTask();
    bool UnRegisterProcTask();
    bool UnInitSmPlugin();
    void UnRegistAliveTask();
    void RegistAliveTask();

private:
    SmPluginMonitor();
    SmPluginMonitor(const SmPluginMonitor &);
    SmPluginMonitor & operator = (const SmPluginMonitor &);

    bool lock {false};

};
}  // namespace chassis
}  // namespace hozon