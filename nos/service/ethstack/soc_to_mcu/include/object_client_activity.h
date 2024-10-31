/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-09-20 11:01:26
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-11-22 14:24:28
 * @FilePath: /nos/service/ethstack/soc_to_mcu/object_client_activity.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: object_client_activity
 */

#ifndef OBJECT_CLIENT_ACTIVITY_H
#define OBJECT_CLIENT_ACTIVITY_H

#include "data_type.h"
#include "hozon/netaos/impl_type_haffusionoutarray.h"
#include "intra_logger.h"

#include "proto/perception/perception_obstacle.pb.h"

namespace hozon {
namespace netaos {
namespace intra {
class ObjectClientActivity {
    using Skeleton = hozon::netaos::v1::skeleton::SocDataServiceSkeleton;

 public:
    ObjectClientActivity();
    ~ObjectClientActivity();
    void Init(std::shared_ptr<Skeleton> skeleton, std::string drivemode);
    void Stop();

 protected:
    std::shared_ptr<Skeleton> skeleton_;
    hozon::netaos::cm::ProtoCMReader<hozon::perception::PerceptionObstacles> reader;

 private:
    uint32_t seq;
    void sendObjectData(const std::shared_ptr<hozon::perception::PerceptionObstacles> Sample);
    void cb(std::shared_ptr<hozon::perception::PerceptionObstacles> msg);
};

}  // namespace intra
}  // namespace netaos
}  // namespace hozon
#endif  // LANES_CLIENT_ACTIVITY_H
