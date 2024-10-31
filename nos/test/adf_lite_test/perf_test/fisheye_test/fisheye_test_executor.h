#include "adf-lite/include/executor.h"

using namespace hozon::netaos::adf_lite;

class FisheyeTestExecutor : public Executor {
public:
    FisheyeTestExecutor();
    ~FisheyeTestExecutor();

    int32_t AlgInit();
    void AlgRelease();

private:
    int32_t hpp_freespace(Bundle* input);
    int32_t avm_stitcher_image(Bundle* input);
    int32_t fisheye_perception_obj_camera(Bundle* input);
    int32_t avm_segmentation_image(Bundle* input);
    int32_t hpp_lane(Bundle* input);
    int32_t slot_det_parking_lot(Bundle* input);
    int32_t ocr_info(Bundle* input);
    int32_t state_machine_1(Bundle* input);
    int32_t planning_test_recvProcess(Bundle* input);
    BaseDataTypePtr GenProtoData(int64_t i);
    template<typename T> BaseDataTypePtr GenData(std::shared_ptr<T> idl_msg);
    
};


REGISTER_ADF_CLASS(FisheyeTest, FisheyeTestExecutor)