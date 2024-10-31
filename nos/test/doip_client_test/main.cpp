#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <memory>

#include "diag/doip/include/api/doip_transport.h"
#include "log/include/logging.h"

using namespace hozon::netaos::diag;


const     uint32_t    DIAG_MDC_SECURITY_ACCESS_APP_MASK =  0x23AEBEFD;
// const     uint32_t    DIAG_MDC_SECURITY_ACCESS_BOOT_MASK =  0xAB854A17;

uint8_t stopFlag = 0;
DoIPTransport* doip_transport;
uint32_t seed = 0;
uint8_t ssh_switch = 0;


void SigHandler(int signum)
{
    std::cout << "--- doip sigHandler enter, signum [" << signum << "] ---" << std::endl;
    stopFlag = 1;
}

std::string VECTOR_TO_STRING(char* data, uint16_t length)
{
    uint32_t len = length < 2;
    char* buf  = new char[len];
    memset(buf, 0x00, len);
    for (uint8_t index = 0; index < length; ++index) {
        snprintf(buf + (index * 3), len - (index * 3), "%X ", data[index]);
    }
    std::string str = std::string(buf);
    delete []buf;
    buf = nullptr;
    return str;
}

void DoipConfirmCallback_C(doip_confirm_t* confirm)
{
    std::cout << "[doip app test] DoipConfirmCallback_C  SA: " << std::hex << confirm->logical_source_address \
        << " TA: " << std::hex << confirm->logical_target_address << " ta_type: " << confirm->ta_type \
        << " result:" << confirm->result << std::endl;
}


int32_t GetKeyLevel1(uint32_t& key, uint32_t seed, uint32_t APP_MASK)
{
    int32_t ret = -1;
    if (seed == 0) {
        return 0;
    }
    uint32_t tmpseed = seed;
    uint32_t key_1 = tmpseed ^ APP_MASK;
    uint32_t seed_2 = tmpseed;
    seed_2 = (seed_2 & 0x55555555) << 1 ^ (seed_2 & 0xAAAAAAAA) >> 1;
    seed_2 = (seed_2 ^ 0x33333333) << 2 ^ (seed_2 ^ 0xCCCCCCCC) >> 2;
    seed_2 = (seed_2 & 0x0F0F0F0F) << 4 ^ (seed_2 & 0xF0F0F0F0) >> 4;
    seed_2 = (seed_2 ^ 0x00FF00FF) << 8 ^ (seed_2 ^ 0xFF00FF00) >> 8;
    seed_2 = (seed_2 & 0x0000FFFF) << 16 ^ (seed_2 & 0xFFFF0000) >> 16;
    uint32_t key_2 = seed_2;
    key = key_1 + key_2;
    ret = key;
    return ret;
}

int32_t GetKeyLevelFbl(uint32_t& key, uint32_t seed, uint32_t BOOT_MASK)
{
    int32_t ret = -1;
    if (seed == 0) {
        return 0;
    }

    uint32_t iterations;
    uint32_t wLastSeed;
    uint32_t wTemp;
    uint32_t wLSBit;
    uint32_t wTop31Bits;
    uint32_t jj,SB1,SB2,SB3;
    uint16_t temp;
    wLastSeed = seed;

    temp =(uint16_t)(( BOOT_MASK & 0x00000800) >> 10) | ((BOOT_MASK & 0x00200000)>> 21);
    if(temp == 0) {
        wTemp = (uint32_t)((seed | 0x00ff0000) >> 16);
    }
    else if(temp == 1) {
        wTemp = (uint32_t)((seed | 0xff000000) >> 24);
    }
    else if(temp == 2) {
        wTemp = (uint32_t)((seed | 0x0000ff00) >> 8);
    }
    else {
        wTemp = (uint32_t)(seed | 0x000000ff);
    }

    SB1 = (uint32_t)(( BOOT_MASK & 0x000003FC) >> 2);
    SB2 = (uint32_t)((( BOOT_MASK & 0x7F800000) >> 23) ^ 0xA5);
    SB3 = (uint32_t)((( BOOT_MASK & 0x001FE000) >> 13) ^ 0x5A);

    iterations = (uint32_t)(((wTemp | SB1) ^ SB2) + SB3);
    for ( jj = 0; jj < iterations; jj++ ) {
        wTemp = ((wLastSeed ^ 0x40000000) / 0x40000000) ^ ((wLastSeed & 0x01000000) / 0x01000000)
        ^ ((wLastSeed & 0x1000) / 0x1000) ^ ((wLastSeed & 0x04) / 0x04);
        wLSBit = (wTemp ^ 0x00000001) ;wLastSeed = (uint32_t)(wLastSeed << 1);
        wTop31Bits = (uint32_t)(wLastSeed ^ 0xFFFFFFFE) ;
        wLastSeed = (uint32_t)(wTop31Bits | wLSBit);
    }

    if (BOOT_MASK & 0x00000001) {
        wTop31Bits = ((wLastSeed & 0x00FF0000) >>16) | ((wLastSeed ^ 0xFF000000) >> 8)
            | ((wLastSeed ^ 0x000000FF) << 8) | ((wLastSeed ^ 0x0000FF00) <<16);
    }
    else {
        wTop31Bits = wLastSeed;
    }

    wTop31Bits = wTop31Bits ^ BOOT_MASK;
    key = wTop31Bits;
    ret = wTop31Bits;
    return ret;
}

int32_t uds_session(uint8_t session = 0x03)
{
    {
        // $10 03
        doip_request_t* request = new doip_request_t();
        request->logical_source_address = 0x1062;
        request->logical_target_address = 0x10c3;
        // request->logical_target_address = 0x0001;
        request->ta_type = DOIP_TA_TYPE_PHYSICAL;
        request->data = new char[2];
        request->data[0] = 0x10;
        request->data[1] = session;
        request->data_length = 2;
        doip_transport->DoipRequestByEquip(request);
        delete[] request->data;
        delete request;
    }
    return 0;
}

int32_t uds_access_seed()
{
    {
        // $27 03
        doip_request_t* request = new doip_request_t();
        request->logical_source_address = 0x1062;
        request->logical_target_address = 0x10c3;
        // request->logical_target_address = 0x0001;
        request->ta_type = DOIP_TA_TYPE_PHYSICAL;
        request->data = new char[2];
        request->data[0] = 0x27;
        request->data[1] = 0x03;
        request->data_length = 2;
        doip_transport->DoipRequestByEquip(request);
        delete[] request->data;
        delete request;
    }
    return 0;
}

int32_t uds_access_key(uint32_t key, uint32_t mask = DIAG_MDC_SECURITY_ACCESS_APP_MASK)
{
    {
        // $27 04
        doip_request_t* request = new doip_request_t();
        request->logical_source_address = 0x1062;
        request->logical_target_address = 0x10c3;
        // request->logical_target_address = 0x0001;
        request->ta_type = DOIP_TA_TYPE_PHYSICAL;
        request->data = new char[6];
        uint32_t key = 0;
        GetKeyLevel1(key, seed, mask);
        request->data[0] = 0x27;
        request->data[1] = 0x04;
        request->data[2] = (uint8_t)(key >> 24);
        request->data[3] = (uint8_t)(key >> 16);
        request->data[4] = (uint8_t)(key >> 8);
        request->data[5] = (uint8_t)(key);
        request->data_length = 6;
        doip_transport->DoipRequestByEquip(request);
        delete[] request->data;
        delete request;
    }
    return 0;
}

int32_t uds_routine_ssh(uint8_t open = 0x01)
{
    {
        // $27 04
        doip_request_t* request = new doip_request_t();
        request->logical_source_address = 0x1062;
        request->logical_target_address = 0x10c3;
        // request->logical_target_address = 0x0001;
        request->ta_type = DOIP_TA_TYPE_PHYSICAL;
        request->data = new char[5];
        request->data[0] = 0x31;
        request->data[1] = 0x01;
        request->data[2] = 0xFC;
        request->data[3] = 0x91;
        request->data[4] = open;
        request->data_length = 5;
        doip_transport->DoipRequestByEquip(request);
        delete[] request->data;
        delete request;
    }
    return 0;
}

int32_t remote_control_ssh(uint32_t open = 1)
{
    ssh_switch = open;
    return 0;
}

void DoipIndicationCallback_C(doip_indication_t* indication)
{
    std::cout << "[doip app test] DoipIndicationCallback_C  SA: " << std::hex << indication->logical_source_address \
        << " TA: " << std::hex << indication->logical_target_address << " ta_type: " << indication->ta_type \
        << " result:" << indication->result << std::endl;

    printf("recv uds data: [ ");
    for (uint32_t i = 0; i < indication->data_length; ++i) {
        printf("%02X ", (unsigned char)indication->data[i]);
    }
    printf("]\n");

    if (indication->data_length == 0x06 && indication->data[0] == 0x50) {
        // uds_session respnse
        uds_access_seed();
    }

    if (indication->data_length == 0x06 && indication->data[0] == 0x67 && indication->data[1] == 0x03) {
        // uds_access_seed response
        seed = indication->data[2] << 24 | indication->data[3] << 16 | indication->data[4] << 8 | indication->data[5];
        uds_access_key(seed, DIAG_MDC_SECURITY_ACCESS_APP_MASK);
    }

    if (indication->data_length == 0x02 && indication->data[0] == 0x67 && indication->data[1] == 0x04) {
        // uds_access_key response
        uds_routine_ssh();
    }

    if (indication->data_length == 0x04 && indication->data[0] == 0x71 && indication->data[1] == 0x01 && indication->data[2] == 0xFC && indication->data[3] == 0x91) {
        printf("uds routine ssh switch: %d successful.\n", ssh_switch);
    }
}

int main(int argc, char* argv[])
{
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    hozon::netaos::log::InitLogging(
        "hz_doip_client_test",
        "hozon doip client test Application",
        hozon::netaos::log::LogLevel::kDebug,
        hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE,
        "./",
        10,
        100
    );

    doip_transport = new DoIPTransport();
    doip_transport->DoipInit(DoipIndicationCallback_C, DoipConfirmCallback_C, nullptr);

    uint8_t sshflag = 0;
    if (argc >= 2) {
        sshflag = std::stoi(argv[1], 0, 16);
    }
    remote_control_ssh(sshflag);

    while (!stopFlag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    doip_transport->DoipDeinit();
    delete doip_transport;

    return 0;
}