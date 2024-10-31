#include "fusion_test_executor.h"
#include "adf-lite/include/ds/builtin_types.h"
#include "proto/test/soc/for_test.pb.h"
#include "idl/generated/chassis.h"
#include "idl/generated/chassisPubSubTypes.h"
#include "fusion_test_logger.h"
#include "idl/generated/freespace.h"
#include "idl/generated/freespacePubSubTypes.h"
#include "idl/generated/lane.h"
#include "idl/generated/lanePubSubTypes.h"
#include "idl/generated/parkinglot.h"
#include "idl/generated/parkinglotPubSubTypes.h"
#include "idl/generated/sensor_ussinfo.h"
#include "idl/generated/sensor_ussinfoPubSubTypes.h"
#include "idl/generated/sensor_uss.h"
#include "idl/generated/sensor_ussPubSubTypes.h"
#include "proto/statemachine/state_machine.pb.h"
#include "test/adf_lite_test/perf_test/util/base.h"

int temp= 0;
std::atomic<uint32_t> g_recv_fisheye_count{0};
uint32_t g_recv_count_old = 0;
std::atomic<uint32_t> g_recv_uss_count{0};
std::atomic<uint32_t> g_total_delta{0};

FusionTestExecutor::FusionTestExecutor() {

}

FusionTestExecutor::~FusionTestExecutor() {

}


int32_t FusionTestExecutor::AlgInit() {
    DsLogger::GetInstance()._logger.Init("FUSI", ADF_LOG_LEVEL_INFO);

    RegistAlgProcessFunc("hpp_freespace", std::bind(&FusionTestExecutor::hpp_freespace_Recv, this, std::placeholders::_1));
    RegistAlgProcessFunc("avm_stitcher_image", std::bind(&FusionTestExecutor::avm_stitcher_image_Recv, this, std::placeholders::_1));
    RegistAlgProcessFunc("fisheye_perception_obj_camera", std::bind(&FusionTestExecutor::fisheye_perception_obj_camera_Recv, this, std::placeholders::_1));
    RegistAlgProcessFunc("avm_segmentation_image", std::bind(&FusionTestExecutor::avm_segmentation_image_Recv, this, std::placeholders::_1));
    RegistAlgProcessFunc("hpp_lane", std::bind(&FusionTestExecutor::hpp_lane_Recv, this, std::placeholders::_1));
    RegistAlgProcessFunc("slot_det_parking_lot", std::bind(&FusionTestExecutor::slot_det_parking_lot_Recv, this, std::placeholders::_1));
    RegistAlgProcessFunc("ocr_info", std::bind(&FusionTestExecutor::ocr_info_Recv, this, std::placeholders::_1));
    RegistAlgProcessFunc("state_machine_1", std::bind(&FusionTestExecutor::state_machine_1_Recv, this, std::placeholders::_1));
    RegistAlgProcessFunc("Object_Info", std::bind(&FusionTestExecutor::Object_Info_Recv, this, std::placeholders::_1));
    RegistAlgProcessFunc("UPA_Info_T", std::bind(&FusionTestExecutor::UPA_Info_T_Recv, this, std::placeholders::_1));
    RegistAlgProcessFunc("UssRawDataSet", std::bind(&FusionTestExecutor::UssRawDataSet_Recv, this, std::placeholders::_1));
    
    return 0;
}

void FusionTestExecutor::AlgRelease() {
}
void FusionTestExecutor::CheckPerfCPU() {
    #define LINE_LENGTH 300
    char line[LINE_LENGTH] = {0};
    FILE *popen_file = popen("ps aux |grep -E 'adf-lite-process' ", "r");
    if(popen_file != NULL) {
        while(fgets(line, LINE_LENGTH, popen_file)!= NULL) {
            FISH_LOG_INFO << "**perf: " << line;
        }
    }
    else {
        return;
    }
    pclose(popen_file);

}

void FusionTestExecutor::CheckTransferTime(BaseDataTypePtr ptr) {
    uint64_t recv_time = GetRealTimestamp_us();
    uint64_t send_time = ptr->__header.timestamp_real_us;
    uint64_t delta = recv_time - send_time;
    g_total_delta += delta;
    if (g_recv_fisheye_count + g_recv_uss_count -  g_recv_count_old > 1000) {
        g_recv_count_old = g_recv_fisheye_count + g_recv_uss_count;
        FISH_LOG_INFO << "**perf: send time is " << send_time << ", receive time is " << recv_time << ", delta is " << delta;
        FISH_LOG_INFO << "**perf: total delta is " << g_total_delta << ", total count is " << g_recv_count_old;

        FISH_LOG_INFO << "**perf: average time is " << double(g_total_delta * 1.0 / g_recv_count_old);
        CheckPerfCPU();
    }
}


int32_t FusionTestExecutor::hpp_freespace_Recv(Bundle* input) {
    BaseDataTypePtr ptr = input->GetOne("hpp_freespace");
    if (!ptr) {
        return -1;
    }
    CheckTransferTime(ptr);
    std::shared_ptr<AlgFreeSpaceOutArray> ego_hmi = std::static_pointer_cast<AlgFreeSpaceOutArray>(ptr->idl_msg);
    temp = temp + uint64_t(ego_hmi.get());
    g_recv_fisheye_count++;

    send_planning_test();


    return 0;
}

int32_t FusionTestExecutor::send_planning_test() {
    std::shared_ptr<AlgUssInfo> ussinfo = std::make_shared<AlgUssInfo>();

    Object_Info obj_info;
    obj_info.wTracker_age(100);
    obj_info.cTracker_ID(100);

    ussinfo->Tracker_Data(obj_info);

    std::array<uint16_t, 12UL> fisheye_count;
    fisheye_count[0] = (uint16_t)(g_recv_fisheye_count >> 16);
    fisheye_count[1] = (uint16_t)g_recv_fisheye_count;
    ussinfo->reserved1(fisheye_count);

    std::array<uint16_t, 12UL> uss_count;
    uss_count[0] = (uint16_t)(g_recv_uss_count >> 16);
    uss_count[1] = (uint16_t)g_recv_uss_count;
    ussinfo->reserved2(uss_count);
    // ussinfo->fisheye_count(g_recv_fisheye_count);
    // ussinfo->uss_count(g_recv_uss_count);

    BaseDataTypePtr data = std::make_shared<BaseData>();
    data->idl_msg = ussinfo;
    SendOutput("planning_test", data);
    return 0;
}

int32_t FusionTestExecutor::avm_stitcher_image_Recv(Bundle* input) {
    BaseDataTypePtr ptr = input->GetOne("avm_stitcher_image");
    if (!ptr) {
        return -1;
    }
    CheckTransferTime(ptr);
    std::shared_ptr<AlgFreeSpaceOutArray> ego_hmi = std::static_pointer_cast<AlgFreeSpaceOutArray>(ptr->idl_msg);
    //FISH_LOG_INFO << "avm_stitcher_image count is " << ego_hmi->count();
    temp = temp + uint64_t(ego_hmi.get());
    g_recv_fisheye_count++;
    return 0;
}


int32_t FusionTestExecutor::fisheye_perception_obj_camera_Recv(Bundle* input) {
    BaseDataTypePtr ptr = input->GetOne("fisheye_perception_obj_camera");
    if (!ptr) {
        return -1;
    }
    CheckTransferTime(ptr);
    std::shared_ptr<AlgLaneDetectionOutArray> ego_hmi = std::static_pointer_cast<AlgLaneDetectionOutArray>(ptr->idl_msg);
    //FISH_LOG_INFO << "fisheye_perception_obj_camera count is " << ego_hmi->count();
    temp = temp + uint64_t(ego_hmi.get());
    g_recv_fisheye_count++;
    return 0;
}
int32_t FusionTestExecutor::avm_segmentation_image_Recv(Bundle* input) {
    BaseDataTypePtr ptr = input->GetOne("avm_segmentation_image");
    if (!ptr) {
        return -1;
    }
    CheckTransferTime(ptr);
    std::shared_ptr<AlgLaneDetectionOutArray> ego_hmi = std::static_pointer_cast<AlgLaneDetectionOutArray>(ptr->idl_msg);
    //FISH_LOG_INFO << "avm_segmentation_image count is " << ego_hmi->count();
    temp = temp + uint64_t(ego_hmi.get());
    g_recv_fisheye_count++;
    return 0;
}
int32_t FusionTestExecutor::hpp_lane_Recv(Bundle* input) {
    BaseDataTypePtr ptr = input->GetOne("hpp_lane");
    if (!ptr) {
        return -1;
    }
    CheckTransferTime(ptr);
    std::shared_ptr<AlgLaneDetectionOutArray> ego_hmi = std::static_pointer_cast<AlgLaneDetectionOutArray>(ptr->idl_msg);
    //FISH_LOG_INFO << "hpp_lane count is " << ego_hmi->count();
    temp = temp + uint64_t(ego_hmi.get());
    g_recv_fisheye_count++;
    return 0;
}
int32_t FusionTestExecutor::slot_det_parking_lot_Recv(Bundle* input) {
    BaseDataTypePtr ptr = input->GetOne("slot_det_parking_lot");
    if (!ptr) {
        return -1;
    }
    CheckTransferTime(ptr);
    std::shared_ptr<AlgParkingLotOutArray> ego_hmi = std::static_pointer_cast<AlgParkingLotOutArray>(ptr->idl_msg);
    //FISH_LOG_INFO << "slot_det_parking_lot count is " << ego_hmi->count();
    temp = temp + uint64_t(ego_hmi.get());
    g_recv_fisheye_count++;
    return 0;
}
int32_t FusionTestExecutor::ocr_info_Recv(Bundle* input) {
    BaseDataTypePtr ptr = input->GetOne("ocr_info");
    if (!ptr) {
        return -1;
    }
    CheckTransferTime(ptr);
    std::shared_ptr<AlgParkingLotOutArray> ego_hmi = std::static_pointer_cast<AlgParkingLotOutArray>(ptr->idl_msg);
    //FISH_LOG_INFO << "ocr_info count is " << ego_hmi->count();
    temp = temp + uint64_t(ego_hmi.get());
    g_recv_fisheye_count++;
    return 0;
}
int32_t FusionTestExecutor::state_machine_1_Recv(Bundle* input) {
    BaseDataTypePtr ptr = input->GetOne("state_machine_1");
    if (!ptr) {
        return -1;
    }
    CheckTransferTime(ptr);
    std::shared_ptr<AlgParkingLotOutArray> ego_hmi = std::static_pointer_cast<AlgParkingLotOutArray>(ptr->idl_msg);
    //FISH_LOG_INFO << "ocr_info count is " << ego_hmi->count();
    temp = temp + uint64_t(ego_hmi.get());
    g_recv_fisheye_count++;
    return 0;
}
int32_t FusionTestExecutor::Object_Info_Recv(Bundle* input) {
    BaseDataTypePtr ptr = input->GetOne("Object_Info");
    if (!ptr) {
        return -1;
    }
    CheckTransferTime(ptr);
    std::shared_ptr<AlgUssInfo> ego_hmi = std::static_pointer_cast<AlgUssInfo>(ptr->idl_msg);
    //FISH_LOG_INFO << "Object_Info count is " << ego_hmi->Tracker_Data().cTracker_ID();
    temp = temp + uint64_t(ego_hmi.get());
    g_recv_uss_count++;
    send_planning_test();
    return 0;
}
int32_t FusionTestExecutor::UPA_Info_T_Recv(Bundle* input) {
    BaseDataTypePtr ptr = input->GetOne("UPA_Info_T");
    if (!ptr) {
        return -1;
    }
    CheckTransferTime(ptr);
    std::shared_ptr<AlgUssRawDataSet> ego_hmi = std::static_pointer_cast<AlgUssRawDataSet>(ptr->idl_msg);
    //FISH_LOG_INFO << "UPA_Info_T count is " << ego_hmi->counter();
    temp = temp + uint64_t(ego_hmi.get());
    g_recv_uss_count++;
    return 0;
}
int32_t FusionTestExecutor::UssRawDataSet_Recv(Bundle* input) {
    BaseDataTypePtr ptr = input->GetOne("UssRawDataSet");
    if (!ptr) {
        return -1;
    }
    CheckTransferTime(ptr);
    std::shared_ptr<AlgUssRawDataSet> ego_hmi = std::static_pointer_cast<AlgUssRawDataSet>(ptr->idl_msg);
    //FISH_LOG_INFO << "UssRawDataSet count is " << ego_hmi->counter();
    temp = temp + uint64_t(ego_hmi.get());
    g_recv_uss_count++;
    return 0;
}

