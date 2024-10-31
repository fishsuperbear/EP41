/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-09-22 09:47:12
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-11-22 14:21:52
 * @FilePath: /nos/service/ethstack/soc_to_mcu/include/ego_mcu_client_activity.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-09-20 11:01:26
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-09-21 20:38:39
 * @FilePath: /nos/service/ethstack/soc_to_mcu/include/lanes_client_activity.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: ego_mcu_client_activity
 */

#ifndef SOCTOMCU_CLIENT_ACTIVITY_H
#define SOCTOMCU_CLIENT_ACTIVITY_H

#include "data_type.h"
#include "hozon/netaos/impl_type_algegotomcuframe.h"
#include "idl/generated/ego2mcu_info.h"
#include "idl/generated/ego2mcu_infoPubSubTypes.h"
#include "intra_logger.h"
namespace hozon {
namespace netaos {
namespace intra {
class EgoToMcuClientActivity {
    using Skeleton = hozon::netaos::v1::skeleton::SocDataServiceSkeleton;

 public:
    EgoToMcuClientActivity();
    ~EgoToMcuClientActivity();
    void Init(std::shared_ptr<Skeleton> skeleton);
    void Stop();

 protected:
    std::shared_ptr<Skeleton> skeleton_;
    hozon::netaos::cm::Proxy egomcu_event_proxy_;

 private:
    void sendEgoToMcuData(const std::shared_ptr<AlgEgo2McuFrame> Sample);
    void cb();
    uint32_t seq;
};

}  // namespace intra
}  // namespace netaos
}  // namespace hozon
#endif  // SOCTOMCU_CLIENT_ACTIVITY_H
