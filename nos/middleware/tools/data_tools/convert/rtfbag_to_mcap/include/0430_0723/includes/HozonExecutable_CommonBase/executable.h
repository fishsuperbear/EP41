/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZONEXECUTABLE_COMMONBASE_EXECUTABLE_H
#define HOZONEXECUTABLE_COMMONBASE_EXECUTABLE_H

#include "ara/adas/adasserviceinterface_proxy.h"
#include "ara/adas/adasserviceinterface_skeleton.h"
#include "hozon/hmi_planningdec/hozoninterface_planningdecisioninfo_proxy.h"
#include "hozon/interface/chassis/hozoninterface_chassis_proxy.h"
#include "hozon/interface/chassis/hozoninterface_chassis_skeleton.h"
#include "hozon/interface/control/hozoninterface_control_proxy.h"
#include "hozon/interface/control/hozoninterface_control_skeleton.h"
#include "hozon/interface/debugcontrol/hozoninterface_debugcontrol_proxy.h"
#include "hozon/interface/debugcontrol/hozoninterface_debugcontrol_skeleton.h"
#include "hozon/interface/debugpb/hozoninterface_debugpb_proxy.h"
#include "hozon/interface/debugpb/hozoninterface_debugpb_skeleton.h"
#include "hozon/interface/debugplan/hozoninterface_debugplan_proxy.h"
#include "hozon/interface/debugplan/hozoninterface_debugplan_skeleton.h"
#include "hozon/interface/ego2mcu/hozoninterface_ego2mcu_proxy.h"
#include "hozon/interface/egohmi/hozoninterface_egohmi_proxy.h"
#include "hozon/interface/egohmi/hozoninterface_egohmi_skeleton.h"
#include "hozon/interface/freespace/hozoninterface_freespace_proxy.h"
#include "hozon/interface/freespace/hozoninterface_freespace_skeleton.h"
#include "hozon/interface/gnssinfo/hozoninterface_gnssinfo_proxy.h"
#include "hozon/interface/gnssinfo/hozoninterface_gnssinfo_skeleton.h"
#include "hozon/interface/imu/hozoninterface_imuinfo_proxy.h"
#include "hozon/interface/imu/hozoninterface_imuinfo_skeleton.h"
#include "hozon/interface/ins/hozoninterface_insinfo_proxy.h"
#include "hozon/interface/ins/hozoninterface_insinfo_skeleton.h"
#include "hozon/interface/laneline/hozoninterface_laneline_proxy.h"
#include "hozon/interface/laneline/hozoninterface_laneline_skeleton.h"
#include "hozon/interface/location/hozoninterface_location_proxy.h"
#include "hozon/interface/location/hozoninterface_location_skeleton.h"
#include "hozon/interface/mapmsg/hozoninterface_mapmsg_proxy.h"
#include "hozon/interface/mapmsg/hozoninterface_mapmsg_skeleton.h"
#include "hozon/interface/mcu2ego/hozoninterface_mcu2ego_skeleton.h"
#include "hozon/interface/objcamera/hozoninterface_obj_camera_proxy.h"
#include "hozon/interface/objcamera/hozoninterface_obj_camera_skeleton.h"
#include "hozon/interface/objfusion/hozoninterface_obj_fusion_proxy.h"
#include "hozon/interface/objfusion/hozoninterface_obj_fusion_skeleton.h"
#include "hozon/interface/objlidar/hozoninterface_obj_lidar_proxy.h"
#include "hozon/interface/objlidar/hozoninterface_obj_lidar_skeleton.h"
#include "hozon/interface/objradar/hozoninterface_obj_radar_proxy.h"
#include "hozon/interface/objradar/hozoninterface_obj_radar_skeleton.h"
#include "hozon/interface/objsignal/hozoninterface_obj_signal_proxy.h"
#include "hozon/interface/objsignal/hozoninterface_obj_signal_skeleton.h"
#include "hozon/interface/parkinglot/hozoninterface_parkinglot_proxy.h"
#include "hozon/interface/parkinglot/hozoninterface_parkinglot_skeleton.h"
#include "hozon/interface/planning/hozoninterface_planning_proxy.h"
#include "hozon/interface/planning/hozoninterface_planning_skeleton.h"
#include "hozon/interface/pointcloud/hozoninterface_pointcloud_proxy.h"
#include "hozon/interface/pointcloud/hozoninterface_pointcloud_skeleton.h"
#include "hozon/interface/prediction/hozoninterface_prediction_proxy.h"
#include "hozon/interface/prediction/hozoninterface_prediction_skeleton.h"
#include "hozon/interface/state_machine/hozoninterface_statemachine_proxy.h"
#include "hozon/interface/state_machine/hozoninterface_statemachine_skeleton.h"
#include "hozon/interface/uss/hozoninterface_uss_proxy.h"
#include "hozon/interface/uss/hozoninterface_uss_skeleton.h"
#include "mdc/cam/camera/cameradecodedmbufserviceinterface_proxy.h"
#include "mdc/cam/camera/cameradecodedmbufserviceinterface_skeleton.h"

#endif // HOZONEXECUTABLE_COMMONBASE_EXECUTABLE_H
