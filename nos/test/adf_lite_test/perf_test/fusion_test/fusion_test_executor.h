#include "adf-lite/include/executor.h"

using namespace hozon::netaos::adf_lite;


class FusionTestExecutor : public Executor {
public:
    FusionTestExecutor();
    ~FusionTestExecutor();

    int32_t AlgInit();
    void AlgRelease();

private:
  void CheckTransferTime(BaseDataTypePtr ptr);
  int32_t hpp_freespace_Recv(Bundle* input);
  int32_t avm_stitcher_image_Recv(Bundle* input);
  int32_t fisheye_perception_obj_camera_Recv(Bundle* input);
  int32_t avm_segmentation_image_Recv(Bundle* input);
  int32_t hpp_lane_Recv(Bundle* input);
  int32_t slot_det_parking_lot_Recv(Bundle* input);
  int32_t ocr_info_Recv(Bundle* input);
  int32_t state_machine_1_Recv(Bundle* input);
  int32_t Object_Info_Recv(Bundle* input);
  int32_t UPA_Info_T_Recv(Bundle* input);
  int32_t UssRawDataSet_Recv(Bundle* input);
  int32_t send_planning_test();
  void CheckPerfCPU();
    
};


REGISTER_ADF_CLASS(FusionTest, FusionTestExecutor)