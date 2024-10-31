/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: The declaration of ThreadPoolHelper
 * Create: 2020-08-17
 */
#ifndef RTF_COM_PROLOC_MANAGER_HELPER_H
#define RTF_COM_PROLOC_MANAGER_HELPER_H
#include <mutex>
#include <set>
#include "rtf/com/types/ros_types.h"
namespace rtf {
namespace com {
namespace utils {
class ProlocManager {
public:
    static std::shared_ptr<ProlocManager>& GetInstance() noexcept;

    void AddProlocService(const vrtf::driver::proloc::ProlocEntityIndex &index) noexcept;

    bool QueryProlocService(const vrtf::driver::proloc::ProlocEntityIndex &index) noexcept;

    void EraseProlocService(const vrtf::driver::proloc::ProlocEntityIndex &index) noexcept;
private:
    std::mutex indexMutex_;
    std::set<vrtf::driver::proloc::ProlocEntityIndex> prolocIndex_;
};
}
}
}
#endif
