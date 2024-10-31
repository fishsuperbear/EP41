
#include <iostream>
#include <vector>

#include "cm/include/method.h"

#include "devm_cm_method.h"
#include "idl/generated/devm.h"
#include "idl/generated/devmPubSubTypes.h"
#include "idl/generated/devmTypeObject.h"

#include "get_read_did.h"

namespace hozon {
namespace netaos {
namespace tools {
using namespace hozon::netaos::cm;

int32_t
ReadDidInfo::StartReadDid()
{
    std::cout << "read-did." << std::endl;
    if(arguments_.size() < 1) {
        std::cout << "parameter err." << std::endl;
        return -1;
    }

    std::shared_ptr<DevmReadDidPubSubType> req_data_type = std::make_shared<DevmReadDidPubSubType>();
    std::shared_ptr<DevmReadDidPubSubType> resp_data_type = std::make_shared<DevmReadDidPubSubType>();
    std::shared_ptr<DevmReadDid> req_data = std::make_shared<DevmReadDid>();
    std::shared_ptr<DevmReadDid> resq_data = std::make_shared<DevmReadDid>();

    Client<DevmReadDid, DevmReadDid> client(req_data_type, resp_data_type);
    client.Init(0, "devm_read_did_tpl");
    
    int32_t online = client.WaitServiceOnline(1000);  //用户需要去调等待服务
    if (online < 0) {
        std::cout << "WaitServiceOnline err." << std::endl;
        client.Deinit();
        return -1;
    }

    // 0.Request
    uint32_t did = strtol(arguments_[0].c_str(), nullptr, 16);
    printf("request  did 0x%04x\n", did);
    req_data->did(did);
    client.Request(req_data, resq_data, 1000);
    printf("response did 0x%x,[%s]\n", resq_data->did(), resq_data->data_value().c_str());

    const char *p = resq_data->data_value().c_str();
    int32_t count = resq_data->data_value().size();
    int32_t line_length = 16;
    int32_t line = count/line_length;
    if (count % line_length != 0) {
        line += 1;
    }
    for (int32_t li = 0; li < line; li++) {
        printf("%08X: ", li*line_length);
        if (li == line - 1) {
            line_length = count % line_length;
        }
        for (int32_t i = 0; i < line_length; i++) {
            printf("%02X ", p[i]);
        }
        printf("| ");
        for (int32_t i = 0; i < line_length; i++) {
            printf("%c", ((p[i] >= 0x20 && p[i]<=0x7E) ? p[i] : '.'));
        }
        printf("\n");
        p += (line_length);

    }

    client.Deinit();
    return 0;
}
}
}
}

