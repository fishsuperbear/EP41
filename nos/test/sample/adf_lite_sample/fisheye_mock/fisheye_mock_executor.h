#include "adf-lite/include/executor.h"
#include "adf/include/log.h"
using namespace hozon::netaos::adf_lite;

class FisheyeMockExecutor : public Executor {
   public:
    FisheyeMockExecutor();
    ~FisheyeMockExecutor();

    int32_t AlgInit();
    void AlgRelease();

   private:
    int32_t FisheyeFrontYield(Bundle* input);
    int32_t FisheyeLeftYield(Bundle* input);
    int32_t FisheyeRightYield(Bundle* input);
    int32_t FisheyeRearYield(Bundle* input);
};

REGISTER_ADF_CLASS(FisheyeMock, FisheyeMockExecutor)