#include "adf-lite/include/executor.h"

using namespace hozon::netaos::adf_lite;


class UssTestExecutor : public Executor {
public:
    UssTestExecutor();
    ~UssTestExecutor();

    int32_t AlgInit();
    void AlgRelease();

private:
  int32_t Object_InfoProcess(Bundle* input);
  int32_t UPA_Info_TProcess(Bundle* input);
  int32_t UssRawDataSetProcess(Bundle* input);
  template<typename T> BaseDataTypePtr GenData(std::shared_ptr<T> idl_msg);
  int32_t planning_test_recvProcess(Bundle* input);
};

REGISTER_ADF_CLASS(UssTest, UssTestExecutor)