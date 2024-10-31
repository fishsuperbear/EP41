/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2024. All rights reserved.
 * Description: https.h is designed for https.
 */
#pragma once

#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <list>

// #include "http_client.h"
#include "entry_point.h"
#include "https_types.h"
#include <atomic>
#include "config_param.h"

namespace hozon {
namespace netaos {
namespace https {

class HttpsImpl;

class Https {
   public:
    static Https& Instance();
    static void Destroy();

    void Init();
    void Deinit();

    int HttpRequest(RequestPtr req_ptr, ResponseHandler handler);
    bool CancelRequest(int id);
    bool IsInited();

   private:
    Https() = default;
    ~Https();

    bool inited_ = false;
    HttpsImpl* impl_ = nullptr;
    static Https* instance_;
};

}
}
}