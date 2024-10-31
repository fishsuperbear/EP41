#include "adf-lite/include/executor.h"
#include "parking_fusion_logger.h"

using namespace hozon::netaos::adf_lite;

class ParkingFusionExecutor : public Executor {
public:
    ParkingFusionExecutor();
    ~ParkingFusionExecutor();

    int32_t AlgInit();
    void AlgRelease();

private:
    int32_t ReceiveCmTopic(Bundle* input);
    int32_t ReceiveFreeData(Bundle* input);
    int32_t ShowLatest();
    int32_t ReceiveWorkFlow1(Bundle* input);
    int32_t ReceiveWorkFlow2(Bundle* input);
    int32_t ReceiveStatusChange(Bundle* input);
    bool _recv_status = true;
};

REGISTER_ADF_CLASS(ParkingFusion, ParkingFusionExecutor)
REGISTER_ADF_CLASS(ParkingFusion2, ParkingFusionExecutor)