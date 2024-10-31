#include <data_type.h>
#include <intra_logger.h>

#include "lanes_client_activity.h"
#include "location_client_activity.h"
#include "object_client_activity.h"
#include "planning_client_activity.h"
#include "state_machine_client_activity.h"

std::mutex mtx;
std::condition_variable cv;
int g_stopFlag = 0;
using namespace hozon::netaos::intra;
using Skeleton = hozon::netaos::v1::skeleton::SocDataServiceSkeleton;

void INTSigHandler(int32_t num) {
    (void)num;
    g_stopFlag = 1;
    std::unique_lock<std::mutex> lck(mtx);
    cv.notify_all();
    std::cout << "Signal Interactive attention" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string mode;
#ifdef BUILD_FOR_MDC
    mode = "./";
#elif BUILD_FOR_ORIN
    mode = "/opt/usr/log/soc_log/";
#else
    mode = "./";
#endif
    hozon::netaos::log::InitLogging("SOC_MCU",                                                             // the id of application
                                    "soc_mcu",                                                             // the log id of application
                                    hozon::netaos::log::LogLevel::kInfo,                                   // the log level of application
                                    hozon::netaos::log::HZ_LOG2FILE,                                       // the output log mode
                                    mode,                                                                  // the log file directory, active when output log to file
                                    10,                                                                    // the max number log file , active when output log to file
                                    20                                                                     // the max size of each  log file , active when output log to file
    );
    ara::core::Initialize();
    std::string str = "1";
    std::string datamode;
    if (argc > 1) {
        datamode = std::string(argv[1]);
    }

    ara::com::InstanceIdentifier instance(str.c_str());
    std::shared_ptr<Skeleton> skeleton_ = std::make_shared<Skeleton>(instance);
    skeleton_->OfferService();

    if (datamode == "test") {
        std::vector<std::thread> works;
        std::thread Task_25Ms([&skeleton_]() {
            while (!g_stopFlag) {
                static int i = 0;
                std::shared_ptr<::hozon::netaos::HafEgoTrajectory> ego_trajectory_data = std::make_shared<::hozon::netaos::HafEgoTrajectory>();
                std::shared_ptr<::hozon::netaos::HafLaneDetectionOutArray> lane_data = std::make_shared<::hozon::netaos::HafLaneDetectionOutArray>();
                std::shared_ptr<::hozon::netaos::HafFusionOutArray> fusion_data = std::make_shared<::hozon::netaos::HafFusionOutArray>();
                std::shared_ptr<::hozon::netaos::APAStateMachineFrame> state_machine_data = std::make_shared<::hozon::netaos::APAStateMachineFrame>();
                std::shared_ptr<::hozon::netaos::AlgEgoToMcuFrame> ego_to_mcu_data = std::make_shared<::hozon::netaos::AlgEgoToMcuFrame>();

                ego_trajectory_data->header.seq = 12345;
                ego_trajectory_data->header.stamp.sec = 56789;
                ego_trajectory_data->header.stamp.nsec = 12345;
                ego_trajectory_data->header.gnssStamp.sec = 56789;
                ego_trajectory_data->header.gnssStamp.nsec = 12345;
                ego_trajectory_data->locSeq = 56789;
                ego_trajectory_data->trajectoryLength = 1.123;
                ego_trajectory_data->trajectoryPeriod = 2.123;
                ego_trajectory_data->proj_heading_offset = 3.123;
                ego_trajectory_data->trajectoryPoint_reference_x = 2.123456;
                ego_trajectory_data->trajectoryPoint_reference_y = 3.123456;

                for (size_t i = 0; i < 60; ++i) {
                    ego_trajectory_data->trajectoryPoints[i].timeRelative = 3.123;
                    ego_trajectory_data->trajectoryPoints[i].x = 2.123;
                    ego_trajectory_data->trajectoryPoints[i].y = 1.123;
                    ego_trajectory_data->trajectoryPoints[i].z = 0.123;
                    ego_trajectory_data->trajectoryPoints[i].theta = 1.123;
                    ego_trajectory_data->trajectoryPoints[i].curvature = 2.123;
                    ego_trajectory_data->trajectoryPoints[i].s = 3.123;
                    ego_trajectory_data->trajectoryPoints[i].speed = 4.123;
                    ego_trajectory_data->trajectoryPoints[i].acc = 5.123;
                }

                ego_trajectory_data->trajectoryValidPointsSize = 1;
                ego_trajectory_data->isEstop = 2;
                ego_trajectory_data->isReplanning = 1;
                ego_trajectory_data->gear = 3;
                ego_trajectory_data->trajectoryType = 4;
                ego_trajectory_data->driviningMode = 5;
                ego_trajectory_data->functionMode = 6;
                ego_trajectory_data->utmzoneID = 16;

                for (size_t i = 0; i < 150; ++i) {
                    ego_trajectory_data->reserve[i] = 66;
                }

                lane_data->header.seq = 12345;
                lane_data->header.stamp.sec = 56789;
                lane_data->header.stamp.nsec = 12345;
                lane_data->header.gnssStamp.sec = 56789;
                lane_data->header.gnssStamp.nsec = 12345;
                lane_data->locSeq = 1234;
                lane_data->isLaneChangeToLeft = 1;
                lane_data->isLaneChangeToRight = 1;

                for (size_t i = 0; i < 8; ++i) {
                    hozon::netaos::HafLaneDetectionOut_A tmp_array;
                    for (size_t j = 0; j < 2; ++j) {
                        hozon::netaos::HafLaneDetectionOut data;
                        data.lanelineSeq = 1234;
                        data.geometryConfidence = 1.123;
                        data.cls = 1;
                        data.typeConfidence = 2.123;
                        data.color = 3;
                        data.colorConfidence = 2.123;
                        data.laneLineWidth = 3.123;
                        data.keyPointVRF.x = 1.123;
                        data.keyPointVRF.y = 2.123;
                        data.keyPointVRF.z = 3.123;
                        data.timeCreation.sec = 1234;
                        data.timeCreation.nsec = 5678;
                        data.laneFit.xStartVRF = 1.123;
                        data.laneFit.xEndVRF = 2.123;
                        data.laneFit.coefficients.a = 1.123;
                        data.laneFit.coefficients.b = 2.123;
                        data.laneFit.coefficients.c = 3.123;
                        data.laneFit.coefficients.d = 4.123;
                        data.laneFit.coefficients.dev_a = 5.123;
                        data.laneFit.coefficients.dev_b = 6.123;
                        data.laneFit.coefficients.dev_c = 7.123;
                        data.laneFit.coefficients.dev_d = 8.123;
                        tmp_array[j] = data;
                    }
                    lane_data->laneDetectionFrontOut[i] = tmp_array;
                }

                for (size_t i = 0; i < 8; ++i) {
                    hozon::netaos::HafLaneDetectionOut_A tmp_array;
                    for (size_t j = 0; j < 2; ++j) {
                        hozon::netaos::HafLaneDetectionOut data;
                        data.lanelineSeq = 1234;
                        data.geometryConfidence = 1.123;
                        data.cls = 1;
                        data.typeConfidence = 2.123;
                        data.color = 3;
                        data.colorConfidence = 2.123;
                        data.laneLineWidth = 3.123;
                        data.keyPointVRF.x = 1.123;
                        data.keyPointVRF.y = 2.123;
                        data.keyPointVRF.z = 3.123;
                        data.timeCreation.sec = 1234;
                        data.timeCreation.nsec = 5678;
                        data.laneFit.xStartVRF = 1.123;
                        data.laneFit.xEndVRF = 2.123;
                        data.laneFit.coefficients.a = 1.123;
                        data.laneFit.coefficients.b = 2.123;
                        data.laneFit.coefficients.c = 3.123;
                        data.laneFit.coefficients.d = 4.123;
                        data.laneFit.coefficients.dev_a = 5.123;
                        data.laneFit.coefficients.dev_b = 6.123;
                        data.laneFit.coefficients.dev_c = 7.123;
                        data.laneFit.coefficients.dev_d = 8.123;
                        tmp_array[j] = data;
                    }
                    lane_data->laneDetectionRearOut[i] = tmp_array;
                }

                fusion_data->header.stamp.sec = 56789;
                fusion_data->header.stamp.nsec = 12345;
                fusion_data->header.gnssStamp.sec = 56789;
                fusion_data->header.gnssStamp.nsec = 12345;
                fusion_data->locSeq = 1234;

                for (size_t i = 0; i < 64; ++i) {
                    ::hozon::netaos::HafFusionOut tmp_data;
                    tmp_data.ObjectID = 1;
                    tmp_data.Type = 2;
                    tmp_data.DetectSensor_Current = 1234;
                    tmp_data.DetectSensor_History = 5678;
                    tmp_data.MaintenanceStatus = 9;
                    tmp_data.TypeConfidence = 8;
                    tmp_data.ExistenceProbability = 7;
                    tmp_data.RectInfo.Center.x = 1.123;
                    tmp_data.RectInfo.Center.y = 2.123;
                    tmp_data.RectInfo.Center.z = 3.123;
                    tmp_data.RectInfo.CenterStdDev.x = 1.123;
                    tmp_data.RectInfo.CenterStdDev.y = 2.123;
                    tmp_data.RectInfo.CenterStdDev.z = 3.123;
                    tmp_data.RectInfo.SizeLWH.x = 1.123;
                    tmp_data.RectInfo.SizeLWH.y = 2.123;
                    tmp_data.RectInfo.SizeLWH.z = 3.123;
                    tmp_data.RectInfo.SizeStdDev.x = 1.123;
                    tmp_data.RectInfo.SizeStdDev.y = 2.123;
                    tmp_data.RectInfo.SizeStdDev.z = 3.123;
                    tmp_data.RectInfo.Orientation = 1.123;
                    tmp_data.RectInfo.OrientationStdDev = 2.123;
                    tmp_data.VelocityAbs.x = 3.123;
                    tmp_data.VelocityAbs.y = 4.123;
                    tmp_data.AccelerationAbs.x = 5.123;
                    tmp_data.AccelerationAbs.y = 6.123;
                    tmp_data.TimeCreation.sec = 4321;
                    tmp_data.TimeCreation.nsec = 5678;
                    tmp_data.MotionPattern = 11;
                    tmp_data.MotionPatternHistory = 22;
                    tmp_data.BrakeLightSt = 33;
                    tmp_data.TurnLightSt = 44;
                    tmp_data.NearSide = 55;
                    tmp_data.Age = 1234;
                    fusion_data->fusionOut[i] = tmp_data;
                }

                state_machine_data->pilot_status.processing_status = 1;
                state_machine_data->pilot_status.camera_status = 2;
                state_machine_data->pilot_status.uss_status = 3;
                state_machine_data->pilot_status.radar_status = 4;
                state_machine_data->pilot_status.lidar_status = 5;
                state_machine_data->pilot_status.velocity_status = 6;
                state_machine_data->pilot_status.perception_status = 7;
                state_machine_data->pilot_status.planning_status = 8;
                state_machine_data->pilot_status.controlling_status = 9;
                state_machine_data->pilot_status.turn_light_status = 10;
                state_machine_data->pilot_status.localization_status = 11;
                state_machine_data->hpp_command.enable_parking_slot_detection = 1;
                state_machine_data->hpp_command.enable_object_detection = 2;
                state_machine_data->hpp_command.enable_freespace_detection = 3;
                state_machine_data->hpp_command.enable_uss = 4;
                state_machine_data->hpp_command.enable_radar = 5;
                state_machine_data->hpp_command.enable_lidar = 6;
                state_machine_data->hpp_command.system_command = 7;
                state_machine_data->hpp_command.emergencybrake_state = 8;
                state_machine_data->hpp_command.system_reset = 9;
                state_machine_data->hpp_command.reserved1 = 10;
                state_machine_data->hpp_command.reserved2 = 11;
                state_machine_data->hpp_command.reserved3 = 12;

                ego_to_mcu_data->header.seq = 12345;
                ego_to_mcu_data->header.stamp.sec = 56789;
                ego_to_mcu_data->header.stamp.nsec = 12345;
                ego_to_mcu_data->header.gnssStamp.sec = 56789;
                ego_to_mcu_data->header.gnssStamp.nsec = 12345;
                ego_to_mcu_data->msg_ego_nnp.NNP_Active_OnOffSet = 1;
                ego_to_mcu_data->msg_ego_nnp.Lanechangeinfor = 2;
                ego_to_mcu_data->msg_ego_nnp.Lanechangedirection = 3;
                ego_to_mcu_data->msg_ego_nnp.LCAudioPlay = 4;
                ego_to_mcu_data->msg_ego_nnp.Lcsndrequest = 5;
                ego_to_mcu_data->msg_ego_nnp.DCLCAudioplay = 6;
                ego_to_mcu_data->msg_ego_nnp.LaneChangeWarning = 7;
                ego_to_mcu_data->msg_ego_nnp.LightRequest = 9;
                ego_to_mcu_data->msg_ego_nnp.NNP_LightRemind = 10;
                ego_to_mcu_data->msg_ego_nnp.lateralCtrtakeover = 1;
                ego_to_mcu_data->msg_ego_nnp.NNP_Scenarios_AudioPlay = 2;
                ego_to_mcu_data->msg_ego_nnp.NNP_Scenarios = 3;
                ego_to_mcu_data->msg_ego_nnp.NNP_RINO_Status = 4;
                ego_to_mcu_data->msg_ego_nnp.PayModeConfirmFeedBack = 5;
                ego_to_mcu_data->msg_ego_nnp.SpdAdaptComfirmFeedback = 6;
                ego_to_mcu_data->msg_ego_nnp.TSR_SpeedSign = 7;
                ego_to_mcu_data->msg_ego_nnp.ALC_mode = 8;
                ego_to_mcu_data->msg_ego_nnp.NNP_indx_HDMapLocationNavi_u8 = 9;
                ego_to_mcu_data->msg_ego_nnp.NNP_indx_CrrntLaneCond_u8 = 10;
                ego_to_mcu_data->msg_ego_nnp.NNP_d_Distance2OnRamp_sg = 1234;
                ego_to_mcu_data->msg_ego_nnp.NNP_d_Distance2DownRamp_sg = 2345;
                ego_to_mcu_data->msg_ego_nnp.NNP_d_DistanceIntoODD_sg = 3456;
                ego_to_mcu_data->msg_ego_nnp.NNP_d_DistanceOutofODD_sg = 4567;
                ego_to_mcu_data->msg_ego_nnp.NNP_d_CrrntLaneWidth_sg = 1.123;
                ego_to_mcu_data->msg_ego_nnp.NNP_crv_CrrntLaneCurve_sg = 2.123;
                ego_to_mcu_data->msg_ego_nnp.NNP_rad_CrrntLaneHead_sg = 3.123;
                ego_to_mcu_data->msg_ego_nnp.NNP_is_NNPMRMFlf_bl = 1;
                ego_to_mcu_data->msg_ego_nnp.NNP_is_NNPMRMDoneFlf_bl = 2;
                ego_to_mcu_data->msg_ego_nnp.NNP_is_NNPEMFlf_bl = 3;
                ego_to_mcu_data->msg_ego_nnp.NNP_is_NNPEMDoneFlf_bl = 4;
                ego_to_mcu_data->msg_ego_nnp.NNP_indx_NNPSoftwareFault_u8 = 5;
                ego_to_mcu_data->msg_ego_nnp.HighBeamReqSt = 6;
                ego_to_mcu_data->msg_ego_nnp.LowBeamReqSt = 7;
                ego_to_mcu_data->msg_ego_nnp.LowHighBeamReqSt = 8;
                ego_to_mcu_data->msg_ego_nnp.HazardLampReqSt = 9;
                ego_to_mcu_data->msg_ego_nnp.HornReqSt = 10;
                ego_to_mcu_data->msg_ego_avp.m_iuss_state_obs = 6;
                ego_to_mcu_data->msg_ego_avp.need_replan_stop = 1;
                ego_to_mcu_data->msg_ego_avp.plan_trigger = 1;
                ego_to_mcu_data->msg_ego_avp.control_enable = 1;
                ego_to_mcu_data->msg_ego_avp.parking_status = 9;
                ego_to_mcu_data->SOC2FCT_TBD_u32_01 = 1;
                ego_to_mcu_data->SOC2FCT_TBD_u32_02 = 2;
                ego_to_mcu_data->SOC2FCT_TBD_u32_03 = 3;
                ego_to_mcu_data->SOC2FCT_TBD_u32_04 = 4;
                ego_to_mcu_data->SOC2FCT_TBD_u32_05 = 5;

                skeleton_->TrajData.Send(*ego_trajectory_data);
                skeleton_->SnsrFsnLaneDate.Send(*lane_data);
                skeleton_->SnsrFsnObj.Send(*fusion_data);
                skeleton_->ApaStateMachine.Send(*state_machine_data);
                skeleton_->AlgEgoToMCU.Send(*ego_to_mcu_data);

                std::cout << "TrajData send:" << i << std::endl;
                i++;
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
        });
        std::thread Task_10Ms([&skeleton_]() {
            while (!g_stopFlag) {
                std::shared_ptr<::hozon::netaos::HafLocation> location_data = std::make_shared<::hozon::netaos::HafLocation>();
                static int i = 0;
                location_data->header.seq = 12345;
                location_data->header.stamp.sec = 56789;
                location_data->header.stamp.nsec = 12345;
                location_data->header.gnssStamp.sec = 56789;
                location_data->header.gnssStamp.nsec = 12345;

                location_data->gpsWeek = 9876;
                location_data->gpsSec = 1.1234;

                location_data->pose.poseWGS.position.x = 4.123456;
                location_data->pose.poseWGS.position.y = 3.123456;
                location_data->pose.poseWGS.position.z = 2.123456;
                location_data->pose.poseWGS.quaternion.x = 1.123;
                location_data->pose.poseWGS.quaternion.y = 2.123;
                location_data->pose.poseWGS.quaternion.z = 3.123;
                location_data->pose.poseWGS.quaternion.w = 4.123;
                location_data->pose.poseWGS.eulerAngle.x = 1.123456;
                location_data->pose.poseWGS.eulerAngle.y = 2.123456;
                location_data->pose.poseWGS.eulerAngle.z = 3.123456;
                location_data->pose.poseWGS.rotationVRF.x = 4.123456;
                location_data->pose.poseWGS.rotationVRF.y = 5.123456;
                location_data->pose.poseWGS.rotationVRF.z = 6.123456;

                location_data->pose.poseLOCAL.position.x = 4.123456;
                location_data->pose.poseLOCAL.position.y = 3.123456;
                location_data->pose.poseLOCAL.position.z = 2.123456;
                location_data->pose.poseLOCAL.quaternion.x = 1.123;
                location_data->pose.poseLOCAL.quaternion.y = 2.123;
                location_data->pose.poseLOCAL.quaternion.z = 3.123;
                location_data->pose.poseLOCAL.quaternion.w = 4.123;
                location_data->pose.poseLOCAL.eulerAngle.x = 1.123456;
                location_data->pose.poseLOCAL.eulerAngle.y = 2.123456;
                location_data->pose.poseLOCAL.eulerAngle.z = 3.123456;
                location_data->pose.poseLOCAL.rotationVRF.x = 4.123456;
                location_data->pose.poseLOCAL.rotationVRF.y = 5.123456;
                location_data->pose.poseLOCAL.rotationVRF.z = 6.123456;

                location_data->pose.poseGCJ02.position.x = 4.123456;
                location_data->pose.poseGCJ02.position.y = 3.123456;
                location_data->pose.poseGCJ02.position.z = 2.123456;
                location_data->pose.poseGCJ02.quaternion.x = 1.123;
                location_data->pose.poseGCJ02.quaternion.y = 2.123;
                location_data->pose.poseGCJ02.quaternion.z = 3.123;
                location_data->pose.poseGCJ02.quaternion.w = 4.123;
                location_data->pose.poseGCJ02.eulerAngle.x = 1.123456;
                location_data->pose.poseGCJ02.eulerAngle.y = 2.123456;
                location_data->pose.poseGCJ02.eulerAngle.z = 3.123456;
                location_data->pose.poseGCJ02.rotationVRF.x = 4.123456;
                location_data->pose.poseGCJ02.rotationVRF.y = 5.123456;
                location_data->pose.poseGCJ02.rotationVRF.z = 6.123456;

                location_data->pose.poseUTM01.position.x = 4.123456;
                location_data->pose.poseUTM01.position.y = 3.123456;
                location_data->pose.poseUTM01.position.z = 2.123456;
                location_data->pose.poseUTM01.quaternion.x = 1.123;
                location_data->pose.poseUTM01.quaternion.y = 2.123;
                location_data->pose.poseUTM01.quaternion.z = 3.123;
                location_data->pose.poseUTM01.quaternion.w = 4.123;
                location_data->pose.poseUTM01.eulerAngle.x = 1.123456;
                location_data->pose.poseUTM01.eulerAngle.y = 2.123456;
                location_data->pose.poseUTM01.eulerAngle.z = 3.123456;
                location_data->pose.poseUTM01.rotationVRF.x = 4.123456;
                location_data->pose.poseUTM01.rotationVRF.y = 5.123456;
                location_data->pose.poseUTM01.rotationVRF.z = 6.123456;

                location_data->pose.poseUTM02.position.x = 4.123456;
                location_data->pose.poseUTM02.position.y = 3.123456;
                location_data->pose.poseUTM02.position.z = 2.123456;
                location_data->pose.poseUTM02.quaternion.x = 1.123;
                location_data->pose.poseUTM02.quaternion.y = 2.123;
                location_data->pose.poseUTM02.quaternion.z = 3.123;
                location_data->pose.poseUTM02.quaternion.w = 4.123;
                location_data->pose.poseUTM02.eulerAngle.x = 1.123456;
                location_data->pose.poseUTM02.eulerAngle.y = 2.123456;
                location_data->pose.poseUTM02.eulerAngle.z = 3.123456;
                location_data->pose.poseUTM02.rotationVRF.x = 4.123456;
                location_data->pose.poseUTM02.rotationVRF.y = 5.123456;
                location_data->pose.poseUTM02.rotationVRF.z = 6.123456;

                location_data->pose.utmZoneID01 = 1234;
                location_data->pose.utmZoneID02 = 5678;

                location_data->velocity.twistVRF.linearVRF.x = 1.123456;
                location_data->velocity.twistVRF.linearVRF.y = 2.123456;
                location_data->velocity.twistVRF.linearVRF.z = 3.123456;
                location_data->velocity.twistVRF.angularVRF.x = 4.123456;
                location_data->velocity.twistVRF.angularVRF.y = 5.123456;
                location_data->velocity.twistVRF.angularVRF.z = 6.123456;

                location_data->acceleration.linearVRF.linearVRF.x = 1.123456;
                location_data->acceleration.linearVRF.linearVRF.y = 2.123456;
                location_data->acceleration.linearVRF.linearVRF.z = 3.123456;
                location_data->acceleration.linearVRF.angularVRF.x = 1.123456;
                location_data->acceleration.linearVRF.angularVRF.y = 2.123456;
                location_data->acceleration.linearVRF.angularVRF.z = 3.123456;

                location_data->coordinateType = 1;
                location_data->rtkStatus = 2;
                location_data->locationState = 3;

                skeleton_->PoseData.Send(*location_data);
                std::cout << "PoseData send:" << i << std::endl;
                i++;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
        while (!g_stopFlag) {
            std::unique_lock<std::mutex> lck(mtx);
            cv.wait(lck);
        }
        if (Task_25Ms.joinable()) {
            Task_25Ms.join();
        }

        if (Task_10Ms.joinable()) {
            Task_10Ms.join();
        }
    } else {
        LocationClientActivity locationActClient;
        PlanningClientActivity planningActClient;
        LanesClientActivity lanesActClient;
        ObjectClientActivity objectActClient;
        StateMachineClientActivity statemachineActClient;
        locationActClient.Init(skeleton_, datamode);
        planningActClient.Init(skeleton_, datamode);
        lanesActClient.Init(skeleton_, datamode);
        objectActClient.Init(skeleton_, datamode);
        statemachineActClient.Init(skeleton_, datamode);
        while (!g_stopFlag) {
            std::unique_lock<std::mutex> lck(mtx);
            cv.wait(lck);
        }

        locationActClient.Stop();
        planningActClient.Stop();
        lanesActClient.Stop();
        objectActClient.Stop();
        statemachineActClient.Stop();
    }
    skeleton_->StopOfferService();
    ara::core::Deinitialize();
    return 0;
}
