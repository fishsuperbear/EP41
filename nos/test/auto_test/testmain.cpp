#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <memory>
#include <mutex>
#include <unistd.h>

#include "uds_request_oncan.h"
#include "uds_request_onip.h"
#include "uds_request_someip.h"
#include "diag_test_cantp.h"
#include "doip_request_test.h"
#include "phm_func_test.h"

#include "common.h"

std::mutex mutex_log_debug;
bool log_debug = false;

#if 0
void printfVecHex(const char *head, uint8_t *value, uint32_t size)
{
    char buf[1024];
    memset(buf, 0, sizeof(buf));
    size = size > ((sizeof(buf)-strlen(head))/3) ? ((sizeof(buf)-strlen(head))/3):size;

    sprintf(buf, "%s", head);
    for (uint32_t i = 0; i < size; i++) {
        sprintf(buf+strlen(head)+i*3, "%02x ", value[i]);
    }
    INFO_LOG << buf;
}
#endif
void printfVecHex(const char *head, uint8_t *value, uint32_t size)
{
    char buf[1024];
    memset(buf,0,sizeof(buf));
    size = size > (1024/3) ? (1024/3):size;
    for (uint32_t i = 0; i < size; i++) {
        snprintf(buf+i*3, sizeof(buf)-i*3, "%02x ", value[i]);
    }

    INFO_LOG << head << buf;
}

int main(int argc, char* argv[])
{
    if (argc >= 2 && strcmp(argv[1], "debug") == 0) {
        log_debug = true;
    }

    UdsResqestFuncOnIP udsRequestOnIP;
    if (udsRequestOnIP.StartTestUdsOnIP() < 0) {
        return -1;
    }

    // UdsResqestFuncOnCan udsRequestOnCan;
    // if (udsRequestOnCan.StartTestUdsOnCan()) {
    //     return -1;
    // }

    // DiagResqestOnCan requestCanTP;
    // if (requestCanTP.StartTestCanTP()) {
    //     return -1;
    // }

    DoipRequestTest doipRequestSocket;
    if (doipRequestSocket.StartTestDoip()) {
        return -1;
    }

    UdsResqestSomeip requestSomeip;
    if (requestSomeip.StartTestUdsSomeIP()) {
        return -1;
    }

    PhmFuncTest cPhmMain;
    if (cPhmMain.AutoTest()) {
        return -1;
    }

    sleep(10);
    return 0;
}
