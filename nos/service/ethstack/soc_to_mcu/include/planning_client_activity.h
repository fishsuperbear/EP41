/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-09-20 11:01:26
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-11-22 14:24:35
 * @FilePath: /nos/service/ethstack/soc_to_mcu/planning_client_activity.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: planning_client_activity
 */

#ifndef PLANNING_CLIENT_ACTIVITY_H
#define PLANNING_CLIENT_ACTIVITY_H
#include "data_type.h"
#include "hozon/netaos/impl_type_algegotomcuframe.h"
#include "hozon/netaos/impl_type_haftrajectorypoint.h"
#include "intra_logger.h"

#include "proto/planning/planning.pb.h"

namespace hozon {
namespace netaos {
namespace intra {

class PlanningClientActivity {
    using Skeleton = hozon::netaos::v1::skeleton::SocDataServiceSkeleton;

 public:
    PlanningClientActivity();
    ~PlanningClientActivity();
    void Init(std::shared_ptr<Skeleton> skeleton, std::string drivemode);
    void Stop();

 protected:
    std::shared_ptr<Skeleton> skeleton_;
    hozon::netaos::cm::ProtoCMReader<hozon::planning::ADCTrajectory> reader;

    /**
     * @brief Set the Uint 8 By Bit
     *
     * @param first_bit first bit
     * @param second_bit second bit
     * @param third_bit third bit
     * @param fourth_bit fourth bit
     * @param fifth_bit fifth bit
     * @param sixth_bit sixth bit
     * @param seventh_bit seventh bit
     * @param eighth_bit eighth bit
     * @return uint8_t
     */
    uint8_t SetUint8ByBit(bool first_bit, bool second_bit, bool third_bit, bool fourth_bit, bool fifth_bit, bool sixth_bit, bool seventh_bit, bool eighth_bit) {
        uint8_t bit8 = 0;
        return (first_bit | (second_bit << 1) | (third_bit << 2) | (fourth_bit << 3) | (fifth_bit << 4) | (sixth_bit << 5) | (seventh_bit << 6) | (third_bit << 7)) | bit8;
    }

 private:
    uint32_t seq;
    void sendPlanningData(const std::shared_ptr<hozon::planning::ADCTrajectory> Sample);
    void sendEgoToMcuData(const hozon::functionmanager::FunctionManagerOut Sample);

    void cb(std::shared_ptr<hozon::planning::ADCTrajectory> msg);
};

}  // namespace intra
}  // namespace netaos
}  // namespace hozon
#endif  // PLANNING_CLIENT_ACTIVITY_H
