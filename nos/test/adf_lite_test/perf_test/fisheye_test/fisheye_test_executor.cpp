#include "fisheye_test_executor.h"
#include "adf-lite/include/ds/builtin_types.h"
#include "proto/test/soc/for_test.pb.h"
#include "idl/generated/freespace.h"
#include "idl/generated/freespacePubSubTypes.h"
#include "idl/generated/lane.h"
#include "idl/generated/lanePubSubTypes.h"
#include "idl/generated/parkinglot.h"
#include "idl/generated/parkinglotPubSubTypes.h"
#include "proto/statemachine/state_machine.pb.h"
#include "idl/generated/sensor_uss.h"
#include "idl/generated/sensor_ussPubSubTypes.h"
#include "idl/generated/sensor_ussinfo.h"
#include "idl/generated/sensor_ussinfoPubSubTypes.h"
#include "test/adf_lite_test/perf_test/util/base.h"
#include "fisheye_test_logger.h"

using namespace std;

FisheyeTestExecutor::FisheyeTestExecutor() {

}

FisheyeTestExecutor::~FisheyeTestExecutor() {

}

int32_t FisheyeTestExecutor::AlgInit() {
    DsLogger::GetInstance()._logger.Init("FISH", ADF_LOG_LEVEL_INFO);

    RegistAlgProcessFunc("hpp_freespace", std::bind(&FisheyeTestExecutor::hpp_freespace, this, std::placeholders::_1));
    RegistAlgProcessFunc("avm_stitcher_image", std::bind(&FisheyeTestExecutor::avm_stitcher_image, this, std::placeholders::_1));
    RegistAlgProcessFunc("fisheye_perception_obj_camera", std::bind(&FisheyeTestExecutor::fisheye_perception_obj_camera, this, std::placeholders::_1));
    RegistAlgProcessFunc("avm_segmentation_image", std::bind(&FisheyeTestExecutor::avm_segmentation_image, this, std::placeholders::_1));
    RegistAlgProcessFunc("hpp_lane", std::bind(&FisheyeTestExecutor::hpp_lane, this, std::placeholders::_1));
    RegistAlgProcessFunc("slot_det_parking_lot", std::bind(&FisheyeTestExecutor::slot_det_parking_lot, this, std::placeholders::_1));
    RegistAlgProcessFunc("ocr_info", std::bind(&FisheyeTestExecutor::ocr_info, this, std::placeholders::_1));
    RegistAlgProcessFunc("state_machine_1", std::bind(&FisheyeTestExecutor::state_machine_1, this, std::placeholders::_1));
    RegistAlgProcessFunc("planning_test_recv", std::bind(&FisheyeTestExecutor::planning_test_recvProcess, this, std::placeholders::_1));

    return 0;
}

void FisheyeTestExecutor::AlgRelease() {
    //FISH_LOG_INFO << "Release fisheye perception.";
}

bool bLog = false;
std::atomic<uint32_t> g_send_count{0};
uint32_t g_send_count_old{0};
bool start_count = false;
bool bOver = false;

int32_t FisheyeTestExecutor::hpp_freespace(Bundle* input) {
    std::shared_ptr<AlgFreeSpaceOutArray> freespace_info = std::make_shared<AlgFreeSpaceOutArray>();

    vector<AlgFreeSpaceOut> freeSpaceOutVector;
	AlgFreeSpaceOut tmp;
	freeSpaceOutVector.emplace_back(tmp);
    freespace_info->freeSpaceOut(freeSpaceOutVector);

    vector<AlgFreeSpace2DOut> freeSpace2DOutVector;
	AlgFreeSpace2DOut tmp2D;
	freeSpace2DOutVector.emplace_back(tmp2D);

    freespace_info->freeSpace2DOut(freeSpace2DOutVector);


    BaseDataTypePtr data = GenData<AlgFreeSpaceOutArray>(freespace_info);
    SendOutput("hpp_freespace", data);
    if(start_count) g_send_count++;

    return 0;
}

int32_t FisheyeTestExecutor::planning_test_recvProcess(Bundle* input) {
    BaseDataTypePtr ptr = input->GetOne("planning_test");
    if (!ptr) {
        return -1;
    }

    std::shared_ptr<AlgUssInfo> ego_hmi = std::static_pointer_cast<AlgUssInfo>(ptr->idl_msg);

    start_count=true;

    if ((g_send_count - g_send_count_old) > 5000) {
        g_send_count_old = g_send_count;
        uint32_t fisheye_count = (ego_hmi->reserved1()[0] << 16) + ego_hmi->reserved1()[1];
        float rate = fisheye_count * 1.0 / g_send_count;
        FISH_LOG_INFO << "fisheye send_count is " << g_send_count << " receive count is " << fisheye_count << " rate is " << rate << " gap is " << (fisheye_count - g_send_count);
    }
    return 0;
}

int32_t FisheyeTestExecutor::avm_stitcher_image(Bundle* input) {
    std::shared_ptr<AlgFreeSpaceOutArray> freespace_info = std::make_shared<AlgFreeSpaceOutArray>();

    vector<AlgFreeSpaceOut> freeSpaceOutVector;
	AlgFreeSpaceOut tmp;
	freeSpaceOutVector.emplace_back(tmp);
    freespace_info->freeSpaceOut(freeSpaceOutVector);

    vector<AlgFreeSpace2DOut> freeSpace2DOutVector;
	AlgFreeSpace2DOut tmp2D;
	freeSpace2DOutVector.emplace_back(tmp2D);

    freespace_info->freeSpace2DOut(freeSpace2DOutVector);


    BaseDataTypePtr data = GenData<AlgFreeSpaceOutArray>(freespace_info);
    SendOutput("avm_stitcher_image", data);
    if(start_count) g_send_count++;

    return 0;
}

int32_t FisheyeTestExecutor::fisheye_perception_obj_camera(Bundle* input) {
    std::shared_ptr<AlgLaneDetectionOutArray> lane_detection_array = std::make_shared<AlgLaneDetectionOutArray>();

    vector<AlgLaneDetectionOut> lane_detectionVector1;
	AlgLaneDetectionOut tmp;
	lane_detectionVector1.emplace_back(tmp);
    vector<vector<AlgLaneDetectionOut>> lane_detectionVector_vector1;
    lane_detectionVector_vector1.emplace_back(lane_detectionVector1);
    lane_detection_array->laneDetectionFrontOut(lane_detectionVector_vector1);

    vector<AlgLaneDetectionOut> lane_detectionVector2;
	AlgLaneDetectionOut tmp2;
	lane_detectionVector2.emplace_back(tmp2);
    vector<vector<AlgLaneDetectionOut>> lane_detectionVector_vector2;
    lane_detectionVector_vector2.emplace_back(lane_detectionVector2);
    lane_detection_array->laneDetectionRearOut(lane_detectionVector_vector2);

    lane_detection_array->count(100);

    BaseDataTypePtr data = GenData<AlgLaneDetectionOutArray>(lane_detection_array);
    SendOutput("fisheye_perception_obj_camera", data);
    if(start_count) g_send_count++;

    return 0;
}

int32_t FisheyeTestExecutor::avm_segmentation_image(Bundle* input) {
    std::shared_ptr<AlgLaneDetectionOutArray> lane_detection_array = std::make_shared<AlgLaneDetectionOutArray>();

    vector<AlgLaneDetectionOut> lane_detectionVector1;
	AlgLaneDetectionOut tmp;
	lane_detectionVector1.emplace_back(tmp);
    vector<vector<AlgLaneDetectionOut>> lane_detectionVector_vector1;
    lane_detectionVector_vector1.emplace_back(lane_detectionVector1);
    lane_detection_array->laneDetectionFrontOut(lane_detectionVector_vector1);

    vector<AlgLaneDetectionOut> lane_detectionVector2;
	AlgLaneDetectionOut tmp2;
	lane_detectionVector2.emplace_back(tmp2);
    vector<vector<AlgLaneDetectionOut>> lane_detectionVector_vector2;
    lane_detectionVector_vector2.emplace_back(lane_detectionVector2);
    lane_detection_array->laneDetectionRearOut(lane_detectionVector_vector2);

    lane_detection_array->count(100);

    BaseDataTypePtr data = GenData<AlgLaneDetectionOutArray>(lane_detection_array);
    SendOutput("avm_segmentation_image", data);
    if(start_count) g_send_count++;

    return 0;
}

int32_t FisheyeTestExecutor::hpp_lane(Bundle* input) {
    std::shared_ptr<AlgLaneDetectionOutArray> lane_detection_array = std::make_shared<AlgLaneDetectionOutArray>();

    vector<AlgLaneDetectionOut> lane_detectionVector1;
	AlgLaneDetectionOut tmp;
	lane_detectionVector1.emplace_back(tmp);
    vector<vector<AlgLaneDetectionOut>> lane_detectionVector_vector1;
    lane_detectionVector_vector1.emplace_back(lane_detectionVector1);
    lane_detection_array->laneDetectionFrontOut(lane_detectionVector_vector1);

    vector<AlgLaneDetectionOut> lane_detectionVector2;
	AlgLaneDetectionOut tmp2;
	lane_detectionVector2.emplace_back(tmp2);
    vector<vector<AlgLaneDetectionOut>> lane_detectionVector_vector2;
    lane_detectionVector_vector2.emplace_back(lane_detectionVector2);
    lane_detection_array->laneDetectionRearOut(lane_detectionVector_vector2);

    lane_detection_array->count(100);

    BaseDataTypePtr data = GenData<AlgLaneDetectionOutArray>(lane_detection_array);
    SendOutput("hpp_lane", data);
    if(start_count) g_send_count++;

    return 0;
}

int32_t FisheyeTestExecutor::slot_det_parking_lot(Bundle* input) {
    std::shared_ptr<AlgParkingLotOutArray> parkinglog_array = std::make_shared<AlgParkingLotOutArray>();

    vector<AlgParkingLotOut> parkinglog;
	AlgParkingLotOut tmp;
	parkinglog.emplace_back(tmp);
    parkinglog_array->parkingLots(parkinglog);

    vector<AlgPathPoint> tracePath;
	AlgPathPoint tmp2;
	tracePath.emplace_back(tmp2);
    parkinglog_array->tracePath(tracePath);

    parkinglog_array->count(100);

    BaseDataTypePtr data = GenData<AlgParkingLotOutArray>(parkinglog_array);
    SendOutput("slot_det_parking_lot", data);
    if(start_count) g_send_count++;

    return 0;
}

int32_t FisheyeTestExecutor::ocr_info(Bundle* input) {
    std::shared_ptr<AlgParkingLotOutArray> parkinglog_array = std::make_shared<AlgParkingLotOutArray>();

    vector<AlgParkingLotOut> parkinglog;
	AlgParkingLotOut tmp;
	parkinglog.emplace_back(tmp);
    parkinglog_array->parkingLots(parkinglog);

    vector<AlgPathPoint> tracePath;
	AlgPathPoint tmp2;
	tracePath.emplace_back(tmp2);
    parkinglog_array->tracePath(tracePath);

    parkinglog_array->count(100);

    BaseDataTypePtr data = GenData<AlgParkingLotOutArray>(parkinglog_array);
    SendOutput("ocr_info", data);
    if(start_count) g_send_count++;

    return 0;
}
int32_t FisheyeTestExecutor::state_machine_1(Bundle* input) {

    std::shared_ptr<AlgParkingLotOutArray> parkinglog_array = std::make_shared<AlgParkingLotOutArray>();

    vector<AlgParkingLotOut> parkinglog;
	AlgParkingLotOut tmp;
	parkinglog.emplace_back(tmp);
    parkinglog_array->parkingLots(parkinglog);

    vector<AlgPathPoint> tracePath;
	AlgPathPoint tmp2;
	tracePath.emplace_back(tmp2);
    parkinglog_array->tracePath(tracePath);

    parkinglog_array->count(100);

    BaseDataTypePtr data = GenData<AlgParkingLotOutArray>(parkinglog_array);
    SendOutput("state_machine_1", data);
    if(start_count) g_send_count++;

    return 0;
}

BaseDataTypePtr FisheyeTestExecutor::GenProtoData(int64_t i) {
    BaseDataTypePtr workflow_result = std::make_shared<BaseData>();

    std::shared_ptr<hozon::state::StateMachine> statemachine_msg(new hozon::state::StateMachine);
    workflow_result->proto_msg = statemachine_msg;

    return workflow_result;
} 

template<typename T>
BaseDataTypePtr FisheyeTestExecutor::GenData(std::shared_ptr<T> idl_msg) {
    BaseDataTypePtr alg_ego_hmi_data = std::make_shared<BaseData>();
    alg_ego_hmi_data->idl_msg = idl_msg;

    alg_ego_hmi_data->__header.timestamp_real_us = GetRealTimestamp_us();
    return alg_ego_hmi_data;

}