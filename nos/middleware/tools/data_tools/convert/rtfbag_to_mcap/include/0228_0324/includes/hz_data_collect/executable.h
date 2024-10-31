/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HZ_DATA_COLLECT_EXECUTABLE_H
#define HZ_DATA_COLLECT_EXECUTABLE_H

#include "hozon/diag/hozoninterface_diagstatusmonitor_proxy.h"
#include "hozon/hmi/hmiadasdataserviceinterface_0x0403_proxy.h"
#include "hozon/hmi/hmiapadasdataserviceinterface_0x0404_proxy.h"
#include "hozon/hmiddsins/hozoninterface_hmiinsinfo_proxy.h"
#include "hozon/hmiddsnns/hozoninterface_hminnsinfo_proxy.h"
#include "hozon/interface/chassis/hozoninterface_chassis_proxy.h"
#include "hozon/interface/datacollect/hozoninterface_datacollect_mcu_skeleton.h"
#include "hozon/interface/datacollect/hozoninterface_datacollect_skeleton.h"
#include "hozon/interface/datacollect/hozoninterface_mcudatacollect_proxy.h"
#include "hozon/interface/datacollect/hozoninterface_mcumaintaindatacollect_proxy.h"
#include "hozon/interface/debugplan/hozoninterface_debugplan_proxy.h"
#include "hozon/interface/ego2mcu/hozoninterface_ego2mcu_proxy.h"
#include "hozon/interface/freespace/hozoninterface_freespace_proxy.h"
#include "hozon/interface/gnssinfo/hozoninterface_gnssinfo_proxy.h"
#include "hozon/interface/imu/hozoninterface_imuinfo_proxy.h"
#include "hozon/interface/laneline/hozoninterface_laneline_proxy.h"
#include "hozon/interface/location/hozoninterface_location_proxy.h"
#include "hozon/interface/location/hozoninterface_locationnodeinfo_proxy.h"
#include "hozon/interface/mapmsg/hozoninterface_mapmsg_proxy.h"
#include "hozon/interface/mcu2ego/hozoninterface_mcu2ego_proxy.h"
#include "hozon/interface/objcamera/hozoninterface_obj_camera_proxy.h"
#include "hozon/interface/objfusion/hozoninterface_obj_fusion_proxy.h"
#include "hozon/interface/objlidar/hozoninterface_obj_lidar_proxy.h"
#include "hozon/interface/objradar/hozoninterface_obj_radar_proxy.h"
#include "hozon/interface/objsignal/hozoninterface_obj_signal_proxy.h"
#include "hozon/interface/parkinglot/hozoninterface_parkinglot_proxy.h"
#include "hozon/interface/planning/hozoninterface_planning_proxy.h"
#include "hozon/interface/pointcloud/hozoninterface_pointcloud_proxy.h"
#include "hozon/interface/prediction/hozoninterface_prediction_proxy.h"
#include "hozon/interface/rawpointcloud/hozoninterface_rawpointcloud_proxy.h"
#include "hozon/interface/uss/hozoninterface_uss_proxy.h"
#include "hozon/soc_mcu/hzsocmcuadasrtfclientserviceinterface_proxy.h"
#include "hozon/soc_mcu/hzsocmcuclientserviceinterface_proxy.h"
#include "mdc/cam/camera/cameradecodedmbufserviceinterface_proxy.h"
#include "mdc/cam/camera/cameraencodedmbufserviceinterface_proxy.h"
#include "mdc/devm/devmc/devmcenterserviceinterface_proxy.h"

#endif // HZ_DATA_COLLECT_EXECUTABLE_H
