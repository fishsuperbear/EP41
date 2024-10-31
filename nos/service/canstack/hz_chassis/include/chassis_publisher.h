/*"Copyright [year] <Copyright Owner>"*/
#pragma once

#include <thread>
#include "can_parser_chassis.h"
#include "cm/include/skeleton.h"
#include "hz_common.h"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "publisher.h"
#include "proto/canbus/chassis.pb.h"
#include "proto/common/header.pb.h"
#include "proto/fsm/function_manager.pb.h"
#include "proto/statemachine/state_machine.pb.h"

namespace hozon {
namespace netaos {
namespace canstack {
namespace chassis {

#define DATA_VALID_TRUE (1)
#define DATA_VALID_FALSE (0)
#define kDEG2RAD (M_PI / 180)

class ChassisPublisher : public hozon::netaos::canstack::Publisher {
   public:
    static ChassisPublisher* Instance();
    virtual ~ChassisPublisher();

    int Init() override;
    void Pub() override;
    int Stop() override;

    void PassCanName(const std::string& canDevice);

   private:
    ChassisPublisher();
    std::string canName_;
    bool serving_;
    int32_t chassis_info_seq_;
    int32_t mcu2ego_info_seq_;
    int32_t mcu2statemachine_info_seq_;
    uint8_t tmp_warning_info_;
    uint8_t tmp_run_state_;
    bool voice_mode_flag_;
    static ChassisPublisher* sinstance_;
    std::thread chassis_info_thread_;

    std::shared_ptr<hozon::netaos::cm::Skeleton> chassis_skeleton_;
    std::shared_ptr<hozon::netaos::cm::Skeleton> mcu2ego_skeleton_;
    std::shared_ptr<hozon::netaos::cm::Skeleton> aeb2ego_skeleton_;
    std::shared_ptr<hozon::netaos::cm::Skeleton> mcu2state_machine_skeleton_;

    int TurnLightHolder(int signal);
    void SetLight(std::shared_ptr<hozon::canbus::Chassis>, std::shared_ptr<hozon::netaos::canstack::chassis::ChassisInfo_t>);
    void SwitchIg(std::shared_ptr<hozon::canbus::Chassis>, uint8_t);
    hozon::canbus::Chassis::GearPosition SwitchGear(const uint8_t);

    /**
 * @brief int转proto枚举
 *
 * @param default_value default_value
 * @param number number value
 */
    template <typename DataType>
    DataType Int2ProtoEnum(const DataType default_value, const int number) {
        DataType enum_data;
        const auto descriptor = google::protobuf::GetEnumDescriptor<DataType>()->FindValueByNumber(number);
        if (descriptor == nullptr || !google::protobuf::internal::ParseNamedEnum(descriptor->type(), descriptor->name(), &enum_data)) {
            enum_data = default_value;
        }
        return enum_data;
    }

    /**
 * @brief uint转proto枚举
 *
 * @param default_value default_value
 * @param number number value
 */
    template <typename DataType>
    DataType Uint2ProtoEnum(const DataType default_value, const uint8_t number) {
        DataType enum_data;
        const auto descriptor = google::protobuf::GetEnumDescriptor<DataType>()->FindValueByNumber(number);
        if (descriptor == nullptr || !google::protobuf::internal::ParseNamedEnum(descriptor->type(), descriptor->name(), &enum_data)) {
            enum_data = default_value;
        }
        return enum_data;
    }

    /**
 * @brief 把can原始轮速方向转换到proto对应的轮速方向
 *
 * @param can_data_dir can数据
 * @return FORWARD = 0;BACKWARD = 1;STANDSTILL = 2;INVALID = 3;
 */
    uint8_t GetWheelDirection(uint8_t can_data_dir) {
        uint8_t type = 0;
        switch (can_data_dir) {
            case 0:
                type = 2;
                break;

            case 1:
                type = 0;
                break;

            case 2:
                type = 1;
                break;

            default:
                type = 0;
                break;
        }
        return type;
    }
};

}  // namespace chassis
}  // namespace canstack
}  // namespace netaos
}  // namespace hozon
