#include "adf-lite/include/core.h"
#include "adf-lite/include/sig_stop.h"

using namespace hozon::netaos::adf_lite;

int main(int argc, char* argv[]) {
    Core core;
    SigHandler& sig_handler = SigHandler::GetInstance();

    core.Start(std::string(argv[1]));   
    sig_handler.Init();

    sig_handler.NeedStopBlocking();
    core.Stop();
}