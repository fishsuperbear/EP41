/*
* Copyright (c) hozonauto. 2021-2021. All rights reserved.
* Description: Http IF class
*/

#pragma once

#include <string>
#include "https_types.h"
#include "http_client.h"
#include "crypto_adapter.h"

namespace hozon {
namespace netaos {
namespace https {

class HttpsImpl {
public:
    HttpsImpl();
    ~HttpsImpl();

    void Init();
    int HttpRequest(RequestPtr req_ptr, ResponseHandler handler);
    bool CancelRequest(int id);
    
private:
    HttpClient http_client_;
};

}
}
}