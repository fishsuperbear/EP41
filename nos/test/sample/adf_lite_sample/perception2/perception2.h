#include "adf-lite/include/executor.h"
#include "adf/include/log.h"

using namespace hozon::netaos::adf_lite;

class Perception2 : public Executor {
public:
    Perception2();
    ~Perception2();

    int32_t AlgInit();
    void AlgRelease();

private:
    int32_t ReceiveWorkFlow1(Bundle* input);
    int32_t ReceiveLinkSample4(Bundle* input, const ProfileToken& token);
};

REGISTER_ADF_CLASS(Perception2, Perception2)