#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <memory>

#include "diag/doip/include/api/doip_transport.h"
#include "log/include/logging.h"

using namespace hozon::netaos::diag;


uint8_t stopFlag = 0;
DoIPTransport* doip_transport;

void SigHandler(int signum)
{
    std::cout << "--- doip sigHandler enter, signum [" << signum << "] ---" << std::endl;
    stopFlag = 1;
}

void DoipIndicationCallback_S(doip_indication_t* indication)
{
    std::cout << "[doip app test] DoipIndicationCallback_S  SA: " << std::hex << indication->logical_source_address \
        << " TA: " << std::hex << indication->logical_target_address << " ta_type: " << indication->ta_type << std::endl;

    printf(" recv uds data is ");
    uint32_t i;
    for (i = 0; i < indication->data_length; ++i) {
        printf("%02X", (unsigned char)indication->data[i]);
    }
    printf("\n");


    doip_request_t* request = new doip_request_t();
    request->logical_source_address = indication->logical_target_address;
    request->logical_target_address = indication->logical_source_address;
    request->ta_type = indication->ta_type;
    request->data = new char[6];
    request->data[0] = 0x50;
    request->data[1] = 0x03;
    request->data[2] = 0x00;
    request->data[3] = 0x01;
    request->data[4] = 0x02;
    request->data[5] = 0x03;
    request->data_length = 6;
    doip_transport->DoipRequestByNode(request);

    delete[] request->data;
    delete request;
}

void DoipRouteCallback_S(doip_route_t* route)
{
    std::cout << "[doip app test] DoipRouteCallback_S  SA: " << std::hex << route->logical_source_address \
        << " logical_target_address: " << std::hex << route->logical_target_address << " ta_type: " << route->ta_type << std::endl;

    printf(" recv uds data is ");
    uint32_t i;
    for (i = 0; i < route->data_length; ++i) {
        printf("%02X", (unsigned char)route->data[i]);
    }
    printf("\n");
}

void DoipConfirmCallback_S(doip_confirm_t* confirm)
{
    std::cout << "[doip app test] DoipConfirmCallback_S  SA: " << std::hex << confirm->logical_source_address \
        << " TA: " << std::hex << confirm->logical_target_address << " ta_type: " << confirm->ta_type << std::endl;
}


int main(int argc, char* argv[])
{
	signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    hozon::netaos::log::InitLogging(
        "hz_doip_server_test",
        "hozon doip server test Application",
        hozon::netaos::log::LogLevel::kDebug,
        hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE,
        "./",
        10,
        100
    );

    doip_transport = new DoIPTransport();
    doip_transport->DoipInit(DoipIndicationCallback_S, DoipConfirmCallback_S, DoipRouteCallback_S);

	while (!stopFlag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    doip_transport->DoipDeinit();
    delete doip_transport;

	return 0;
}