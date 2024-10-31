#include "adf-lite/include/executor.h"
#include "adf/include/log.h"

using namespace hozon::netaos::adf_lite;

class Perception1 : public Executor {
public:
    Perception1();
    ~Perception1();

    int32_t AlgInit();
    void AlgRelease();

private:
    int32_t ReceiveWorkFlow1(Bundle* input);
    int32_t ReceiveLinkSample2(Bundle* input, const ProfileToken& token);
};

REGISTER_ADF_CLASS(Perception1, Perception1)