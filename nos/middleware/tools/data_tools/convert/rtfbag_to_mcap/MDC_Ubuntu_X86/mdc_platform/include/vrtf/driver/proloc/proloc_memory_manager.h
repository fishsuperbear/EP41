/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: handle DDSDriver to service find and sub pub
 * Create: 2019-07-24
 */
#ifndef VRTF_VCC_PROLOC_H
#define VRTF_VCC_PROLOC_H
#include <mutex>
#include "vrtf/vcc/api/types.h"
#include "vrtf/vcc/utils/thread_pool.h"
#include "vrtf/driver/proloc/proloc_driver_types.h"
namespace vrtf {
namespace driver {
namespace proloc {
class ProlocMemoryManager {
public:
    ProlocMemoryManager() = default;
    virtual ~ProlocMemoryManager() = default;
    ProlocMemoryManager(const ProlocMemoryManager& other) = delete;
    ProlocMemoryManager& operator=(ProlocMemoryManager const &prolocMemoryManager) = delete;
    virtual void EnableEventClient(const ProlocEntityIndex& index, ClientUid const clientUid, const size_t cacheSize,
        const bool isField, vrtf::vcc::api::types::EventReceiveHandler handler) = 0;
    virtual std::vector<std::uint8_t*> ReadProlocEvent(
            const vrtf::driver::proloc::ProlocEntityIndex& index, ClientUid const clientUid, const std::int32_t size) = 0;
    virtual void ReturnLoan(
            const vrtf::driver::proloc::ProlocEntityIndex& index, ClientUid const clientUid, const std::uint8_t* data) = 0;
    virtual void SetReceiveHandler(vrtf::vcc::api::types::EventReceiveHandler handler, ClientUid const clientUid,
                                   ProlocEntityIndex index) = 0;
    virtual void UnSubscribeClient(const ProlocEntityIndex& index, ClientUid const clientUid) = 0;
    virtual size_t GetMessageNumber(const vrtf::driver::proloc::ProlocEntityIndex& index,
                                    ClientUid const clientUid) noexcept = 0;
};
using MethodReceiveHandler = std::function<void(std::shared_ptr<vrtf::driver::proloc::ProlocMethodMsg>)>;
class ProlocMethodManager {
public:
    ProlocMethodManager() = default;
    virtual ~ProlocMethodManager() = default;
    ProlocMethodManager(const ProlocMethodManager& other) = delete;
    ProlocMethodManager& operator=(ProlocMethodManager const &prolocMemoryManager) = delete;

    virtual void ReturnLoan(const std::uint8_t* data, const ProlocEntityIndex &index) = 0;
    virtual void UnRegisterMethod(const ProlocEntityIndex &index) = 0;
};
}
}
}
#endif // INC_ARA_VCC_DRIVER_DDS_DRIVER_HPP_
