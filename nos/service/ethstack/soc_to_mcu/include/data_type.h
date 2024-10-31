/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-09-20 11:03:54
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-09-22 14:49:56
 * @FilePath: /nos/service/ethstack/soc_to_mcu/include/data_type.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: data tpye define
 */

#ifndef DATA_TYPE_H
#define DATA_TYPE_H
#include <time.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include <ara/core/initialization.h>

#include "cm/include/method.h"
#include "cm/include/proto_cm_reader.h"
#include "cm/include/proxy.h"
#include "cm/include/skeleton.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "hozon/netaos/v1/socdataservice_skeleton.h"

#include "proto/common/header.pb.h"
#include "proto/common/types.pb.h"
namespace hozon {
namespace netaos {
namespace intra {

const uint16_t faultObj = 1;
const uint16_t reportStatus = 1;
// data max length
const uint32_t stringSize = 20;
const uint32_t trajectoryPointsLength = 60;
const uint32_t covarianceLength = 6 * 6;
const uint32_t keyPointVRFLength = 1;
const uint32_t laneDetectionFrontOutLength = 8;
const uint32_t laneDetectionFrontOutLength1 = 2;
const uint32_t laneDetectionRearOutLength = 8;
const uint32_t laneDetectionRearOutLength1 = 2;
const uint32_t cornersLength = 32;
const uint32_t sensorIDLength = 32;
const uint32_t fusionOutLength = 64;
const uint32_t Uint8Length = 1400;
const uint32_t HeaderLength = 4;
const uint32_t BodyLength = 1396;
const int32_t LaneIndexError = 255;

enum TapilotMode {
    no_control,
    NNP,
    AVP,
};
const std::string nnplanetopic = "/perception/fsd/transportelement_1";
const std::string hpplanetopic = "/perception/parking/transportelement_2";
const std::string nnplocationtopic = "/localization/location";
const std::string hpplocationtopic = "/perception/parking/slam_location";
const std::string nnpobjecttopic = "/perception/fsd/obj_fusion_1";
const std::string hppobjecttopic = "/perception/parking/obj_fusion_2";
const std::string planningtopic = "/planning/ego_trajectory";
const std::string statemachinetopic = "sm_to_mcu";
const std::string soctomcutopic = "ego2mcu";

}  // namespace intra
}  // namespace netaos
}  // namespace hozon
#endif  // DATA_TYPE_H
