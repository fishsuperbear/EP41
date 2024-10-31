#include "adf-lite/include/executor.h"
#include "parking_fusion_logger.h"

using namespace hozon::netaos::adf_lite;

class MyTeset {
public:
    ParkingFusionExecutor();
    ~ParkingFusionExecutor();

    int32_t AlgInit();
    void AlgRelease();

private:
    int32_t ReceiveCmTopic(Bundle* input);
    int32_t ReceiveFreeData(Bundle* input);
    int32_t ShowLatest();
};

// std::string GetExecutorTag() {
//     return Executor::GetTag();
// }

REGISTER_ADF_CLASS("ParkingFusion", ParkingFusionExecutor)