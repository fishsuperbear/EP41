#pragma once
#include <cmath>
#include <iostream>
#include "adf/include/log.h"
#include "impl_type_idt_hpp_info_struct.h"
#include "impl_type_idt_hpp_location_struct.h"
#include "impl_type_idt_hpp_mapobjectdisplay_struct.h"
#include "impl_type_idt_hpp_path_struct.h"
#include "impl_type_idt_ins_info_struct.h"
#include "impl_type_idt_nns_info_struct.h"

#include "si_adasdataservice_skeleton.h"
#include "si_apadataservice_skeleton.h"

namespace hozon {
namespace netaos {
namespace extra {

void TestAdasData(std::shared_ptr<ap_datatype_package::datatypes::IDT_ADAS_Dataproperties_Struct> adas_data) {
    adas_data->timestamp = 0x1122334455667788;

    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    uint64_t ns = time.tv_nsec / 1000;
    adas_data->locFusionInfo.allFusionPosResult.ticktime = time.tv_sec * 1000 + ns;
    std::cout << "adas_data->locFusionInfo.allFusionPosResult.ticktime = " << adas_data->locFusionInfo.allFusionPosResult.ticktime << std::endl;
    adas_data->locFusionInfo.allFusionPosResult.status = 'A';
    adas_data->locFusionInfo.allFusionPosResult.ns = 'N';
    adas_data->locFusionInfo.allFusionPosResult.ew = 'E';
    adas_data->locFusionInfo.allFusionPosResult.fusiontype = 1;
    adas_data->locFusionInfo.allFusionPosResult.posEnu_Longitude = 120.12345;
    adas_data->locFusionInfo.allFusionPosResult.posEnu_latitude = 30.12345;
    adas_data->locFusionInfo.allFusionPosResult.speed = 100.12345;
    adas_data->locFusionInfo.allFusionPosResult.course = 101.12345;
    adas_data->locFusionInfo.allFusionPosResult.alt = 102.12345;
    adas_data->locFusionInfo.allFusionPosResult.posAcc = 103.12345;
    adas_data->locFusionInfo.allFusionPosResult.courseAcc = 104.12345;
    adas_data->locFusionInfo.allFusionPosResult.altAcc = 105.12345;
    adas_data->locFusionInfo.allFusionPosResult.speedAcc = 106.12345;
    adas_data->locFusionInfo.allFusionPosResult.datetime = 0x1122334455667788;
    adas_data->locFusionInfo.laneFusionResult.ticktime = 0x1122334455667788;
    adas_data->locFusionInfo.laneFusionResult.indices = 1;
    adas_data->locFusionInfo.laneFusionResult.probs = 2.12345;
    adas_data->locFusionInfo.laneFusionResult.lateralOffsetLeft = 3.12345;
    adas_data->locFusionInfo.laneFusionResult.lateralOffsetLeftAcc = 4.12345;
    adas_data->locFusionInfo.laneFusionResult.lateralOffsetRight = 5.12345;
    adas_data->locFusionInfo.laneFusionResult.lateralOffsetRightAcc = 6.12345;

    ap_datatype_package::datatypes::IDT_PosCoordLocal_struct tmp_decision;
    tmp_decision.posCoordLoca_X = 1.12345;
    tmp_decision.posCoordLoca_Y = 2.12345;
    tmp_decision.posCoordLoca_Z = 3.12345;
    for (int i = 0; i < 64; i++) {
        adas_data->decisionInfo[i] = tmp_decision;
    
    }

    adas_data->functionstate.laneChangedStatus = 0;
    adas_data->functionstate.laneChangedType = 1;
    adas_data->functionstate.drivemode = 2;
    adas_data->functionstate.padding_u8_1 = 4;

    ap_datatype_package::datatypes::IDT_DynamicSRObject_Struct tmp_dyna;
    tmp_dyna.id = 1;
    tmp_dyna.type = 2;
    tmp_dyna.brakeLightStatus = 3;
    tmp_dyna.carLightStatus = 4;
    tmp_dyna.localPose.posCoordLoca_X = 1.12345;
    tmp_dyna.localPose.posCoordLoca_Y = 2.12345;
    tmp_dyna.localPose.posCoordLoca_Z = 3.12345;
    tmp_dyna.heading = 5.12345;
    tmp_dyna.obSize.obSize_length = 1.12345;
    tmp_dyna.obSize.obSize_width = 2.12345;
    tmp_dyna.obSize.obSize_height = 3.12345;
    tmp_dyna.isHighlight = 6;
    for (int i = 0; i < 64; i++) {
        adas_data->dynamicSRData[i] = tmp_dyna;
    }

   ap_datatype_package::datatypes::IDT_StaticSRObject_Struct tmp_static;
    tmp_static.id = 1;
    tmp_static.type = 2;
    tmp_static.localPose.posCoordLoca_X = 1.12345;
    tmp_static.localPose.posCoordLoca_Y = 2.12345;
    tmp_static.localPose.posCoordLoca_Z = 3.12345;
    for (int i = 0; i < 16; i++) {
        adas_data->staticSRData[i] = tmp_static;
    }

    ap_datatype_package::datatypes::IDT_LaneData_Struct tmp_lane;
    tmp_lane.lane_state = 1;
    tmp_lane.lane_color = 2;
    tmp_lane.lane_type = 3;
    tmp_lane.lane_ID = 4;

    tmp_lane.lane_equation_C0 = 5.12345;
    tmp_lane.lane_equation_C1 = 6.12345;
    tmp_lane.lane_equation_C2 = 7.12345;
    tmp_lane.lane_equation_C3 = 8.12345;
    tmp_lane.laneWidth = 9.12345;
    tmp_lane.laneLineWidth = 10.12345;
    tmp_lane.lane_start_X = 11.12345;
    tmp_lane.lane_start_Y = 12.12345;
    tmp_lane.lane_end_X = 13.12345;
    tmp_lane.lane_end_Y = 14.12345;
    for (int i = 0; i < 8; i++) {
        adas_data->laneData[i] = tmp_lane;
    }

    ap_datatype_package::datatypes::IDT_IMUdata_Struct_ref tmp_imudata;
    tmp_imudata.angularVelocity.imuPoint_x = 8.12345;
    tmp_imudata.angularVelocity.imuPoint_y = 9.12345;
    tmp_imudata.angularVelocity.imuPoint_z = 10.12345;
    tmp_imudata.linearAcceleration.imuPoint_x = 8.12345;
    tmp_imudata.linearAcceleration.imuPoint_y = 9.12345;
    tmp_imudata.linearAcceleration.imuPoint_z = 10.12345;
    adas_data->imudata = tmp_imudata;

}
void TestApaData(std::shared_ptr<ap_datatype_package::datatypes::IDT_APA_Dataproperties_Struct> apa_data) {
    apa_data->timestamp = 0x1122334455667788;

    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    uint64_t ns = time.tv_nsec / 1000;
    apa_data->locFusionInfo.allFusionPosResult.ticktime = time.tv_sec * 1000 + ns;
    std::cout << "apa_data->locFusionInfo.allFusionPosResult.ticktime = " << apa_data->locFusionInfo.allFusionPosResult.ticktime << std::endl;
    apa_data->locFusionInfo.allFusionPosResult.status = 'A';
    apa_data->locFusionInfo.allFusionPosResult.ns = 'N';
    apa_data->locFusionInfo.allFusionPosResult.ew = 'E';
    apa_data->locFusionInfo.allFusionPosResult.fusiontype = 1;
    apa_data->locFusionInfo.allFusionPosResult.posEnu_Longitude = 120.12345;
    apa_data->locFusionInfo.allFusionPosResult.posEnu_latitude = 30.12345;
    apa_data->locFusionInfo.allFusionPosResult.speed = 100.12345;
    apa_data->locFusionInfo.allFusionPosResult.course = 101.12345;
    apa_data->locFusionInfo.allFusionPosResult.alt = 102.12345;
    apa_data->locFusionInfo.allFusionPosResult.posAcc = 103.12345;
    apa_data->locFusionInfo.allFusionPosResult.courseAcc = 104.12345;
    apa_data->locFusionInfo.allFusionPosResult.altAcc = 105.12345;
    apa_data->locFusionInfo.allFusionPosResult.speedAcc = 106.12345;
    apa_data->locFusionInfo.allFusionPosResult.datetime = 0x1122334455667788;

    apa_data->locFusionInfo.laneFusionResult.ticktime = 0x1122334455667788;
    apa_data->locFusionInfo.laneFusionResult.indices = 1;
    apa_data->locFusionInfo.laneFusionResult.probs = 2.12345;
    apa_data->locFusionInfo.laneFusionResult.lateralOffsetLeft = 3.12345;
    apa_data->locFusionInfo.laneFusionResult.lateralOffsetLeftAcc = 4.12345;
    apa_data->locFusionInfo.laneFusionResult.lateralOffsetRight = 5.12345;
    apa_data->locFusionInfo.laneFusionResult.lateralOffsetRightAcc = 6.12345;


    ap_datatype_package::datatypes::IDT_DynamicSRObject_Struct tmp_dyna;
    tmp_dyna.id = 1;
    tmp_dyna.type = 2;
    tmp_dyna.brakeLightStatus = 3;
    tmp_dyna.carLightStatus = 4;
    tmp_dyna.localPose.posCoordLoca_X = 1.12345;
    tmp_dyna.localPose.posCoordLoca_Y = 2.12345;
    tmp_dyna.localPose.posCoordLoca_Z = 3.12345;
    tmp_dyna.heading = 5.12345;
    tmp_dyna.obSize.obSize_length = 1.12345;
    tmp_dyna.obSize.obSize_width = 2.12345;
    tmp_dyna.obSize.obSize_height = 3.12345;
    tmp_dyna.isHighlight = 6;
    for (int i = 0; i < 64; i++) {
        apa_data->dynamicSRData[i] = tmp_dyna;
    }

   ap_datatype_package::datatypes::IDT_StaticSRObject_Struct tmp_static;
    tmp_static.id = 1;
    tmp_static.type = 2;
    tmp_static.localPose.posCoordLoca_X = 1.12345;
    tmp_static.localPose.posCoordLoca_Y = 2.12345;
    tmp_static.localPose.posCoordLoca_Z = 3.12345;
    for (int i = 0; i < 16; i++) {
        apa_data->staticSRData[i] = tmp_static;
    }

    ap_datatype_package::datatypes::IDT_LaneData_Struct tmp_lane;
    tmp_lane.lane_state = 1;
    tmp_lane.lane_color = 2;
    tmp_lane.lane_type = 3;
    tmp_lane.lane_ID = 4;

    tmp_lane.lane_equation_C0 = 5.12345;
    tmp_lane.lane_equation_C1 = 6.12345;
    tmp_lane.lane_equation_C2 = 7.12345;
    tmp_lane.lane_equation_C3 = 8.12345;
    tmp_lane.laneWidth = 9.12345;
    tmp_lane.laneLineWidth = 10.12345;
    tmp_lane.lane_start_X = 11.12345;
    tmp_lane.lane_start_Y = 12.12345;
    tmp_lane.lane_end_X = 13.12345;
    tmp_lane.lane_end_Y = 14.12345;
    for (int i = 0; i < 8; i++) {
        apa_data->laneData[i] = tmp_lane;
    }

    ap_datatype_package::datatypes::IDT_IMUdata_Struct_ref tmp_imudata;
    tmp_imudata.angularVelocity.imuPoint_x = 8.12345;
    tmp_imudata.angularVelocity.imuPoint_y = 9.12345;
    tmp_imudata.angularVelocity.imuPoint_z = 10.12345;
    tmp_imudata.linearAcceleration.imuPoint_x = 8.12345;
    tmp_imudata.linearAcceleration.imuPoint_y = 9.12345;
    tmp_imudata.linearAcceleration.imuPoint_z = 10.12345;

    apa_data->imudata = tmp_imudata;
}
}  // namespace extra
}  // namespace netaos
}  // namespace hozon