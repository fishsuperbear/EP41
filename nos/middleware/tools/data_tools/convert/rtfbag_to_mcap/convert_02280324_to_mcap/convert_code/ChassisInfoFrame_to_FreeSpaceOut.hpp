

#pragma once
#include "hozon/chassis/impl_type_chassisinfoframe.h"  //mdc 数据变量
#include "proto/perception/perception_freespace.pb.h"  //proto 数据变量

hozon::perception::datacollection::FreeSpaceOutArray ChassisInfoFrameToFreeSpaceOut(hozon::chassis::ChassisInfoFrame mdc_data) {
    hozon::perception::datacollection::FreeSpaceOutArray proto_data;
    proto_data.set_count(1);
    return proto_data;
}