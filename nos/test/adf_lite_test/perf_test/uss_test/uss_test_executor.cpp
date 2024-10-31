#include "uss_test_executor.h"
#include "adf-lite/include/ds/builtin_types.h"
#include "proto/test/soc/for_test.pb.h"
#include "idl/generated/sensor_ussinfo.h"
#include "idl/generated/sensor_ussinfoPubSubTypes.h"
#include "idl/generated/sensor_uss.h"
#include "idl/generated/sensor_ussPubSubTypes.h"
#include "uss_test_logger.h"
#include "test/adf_lite_test/perf_test/util/base.h"

bool bLog = false;
std::atomic<uint16_t> g_send_uss_count{0};
uint16_t g_send_uss_count_old{0};
bool start_count = false;
bool bOver = false;

UssTestExecutor::UssTestExecutor() {

}

UssTestExecutor::~UssTestExecutor() {

}
int32_t UssTestExecutor::AlgInit() {
    DsLogger::GetInstance()._logger.Init("USS", ADF_LOG_LEVEL_INFO);

    RegistAlgProcessFunc("Object_Info", std::bind(&UssTestExecutor::Object_InfoProcess, this, std::placeholders::_1));
    RegistAlgProcessFunc("UPA_Info_T", std::bind(&UssTestExecutor::UPA_Info_TProcess, this, std::placeholders::_1));
    RegistAlgProcessFunc("UssRawDataSet", std::bind(&UssTestExecutor::UssRawDataSetProcess, this, std::placeholders::_1));
    RegistAlgProcessFunc("planning_test_recv", std::bind(&UssTestExecutor::planning_test_recvProcess, this, std::placeholders::_1));
    return 0;
}

void UssTestExecutor::AlgRelease() {
}

int32_t UssTestExecutor::Object_InfoProcess(Bundle* input) {
    std::shared_ptr<AlgUssInfo> ussinfo = std::make_shared<AlgUssInfo>();

    Object_Info obj_info;
    obj_info.wTracker_age(10);
    obj_info.cTracker_ID(10);

    ussinfo->Tracker_Data(obj_info);

    BaseDataTypePtr data = GenData<AlgUssInfo>(ussinfo);
    SendOutput("Object_Info", data);
    if(start_count) g_send_uss_count++;

    return 0;
}

int32_t UssTestExecutor::UPA_Info_TProcess(Bundle* input) {
    
    std::shared_ptr<AlgUssRawDataSet> ussrawdata_set = std::make_shared<AlgUssRawDataSet>();
    AlgUssRawData_APA ussrawdata;
	ussrawdata.counter(10);
    ussrawdata.distance(1000);

    ussrawdata_set->fls_info(ussrawdata);


    BaseDataTypePtr data = GenData<AlgUssRawDataSet>(ussrawdata_set);
    SendOutput("UPA_Info_T", data);
    if(start_count) g_send_uss_count++;

    return 0;
}



int32_t UssTestExecutor::UssRawDataSetProcess(Bundle* input) {
    std::shared_ptr<AlgUssRawDataSet> ussrawdata_set = std::make_shared<AlgUssRawDataSet>();
    AlgUssRawData_APA ussrawdata;
	ussrawdata.counter(10);
    ussrawdata.distance(1000);

    ussrawdata_set->fls_info(ussrawdata);


    BaseDataTypePtr data = GenData<AlgUssRawDataSet>(ussrawdata_set);
    SendOutput("UssRawDataSet", data);
    if(start_count) g_send_uss_count++;

    return 0;
}

int32_t UssTestExecutor::planning_test_recvProcess(Bundle* input) {
    BaseDataTypePtr ptr = input->GetOne("planning_test");
    if (!ptr) {
        return -1;
    }
    std::shared_ptr<AlgUssInfo> ego_hmi = std::static_pointer_cast<AlgUssInfo>(ptr->idl_msg);
    start_count=true;

    if ((g_send_uss_count - g_send_uss_count_old) > 10000) {
        g_send_uss_count_old = g_send_uss_count;
        uint32_t uss_count = (ego_hmi->reserved2()[0] << 16) + ego_hmi->reserved2()[1];
        float rate = uss_count * 1.0 / g_send_uss_count;
        FISH_LOG_INFO << "uss send_count is " << g_send_uss_count << " receive count is " << uss_count << " rate is " << rate << " gap is " << (uss_count - g_send_uss_count);
    }
    return 0;
}

template<typename T>
BaseDataTypePtr UssTestExecutor::GenData(std::shared_ptr<T> idl_msg) {
    BaseDataTypePtr alg_ego_hmi_data = std::make_shared<BaseData>();
    alg_ego_hmi_data->idl_msg = idl_msg;
    
    alg_ego_hmi_data->__header.timestamp_real_us = GetRealTimestamp_us();
    return alg_ego_hmi_data;
}