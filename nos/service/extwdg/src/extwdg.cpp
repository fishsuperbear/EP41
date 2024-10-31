/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: main function definition
 */

#include "extwdg.h"
#include "extwdg_logger.h"

namespace hozon {
namespace netaos {
namespace extwdg {

ExtWdg::ExtWdg(std::vector<TransportInfo> transinfo)
{
    // TransportInfo info;
    // trans_info.emplace_back(std::move(info));
    trans_info = transinfo;
    transport_ = std::make_shared<Transport>(trans_info);
}

int32_t ExtWdg::Init()
{
    EW_INFO << "ExtWdg::Init() enter!";
    if(init_) {
        EW_INFO << "ExtWdg::Init() has been entered!";
        return 0;
    }

    if(transport_->Init() != 0) {
        EW_INFO << "transport_->Init() failed!";
        return -1;
    }
    return 0;
}

void
ExtWdg::DeInit()
{
    init_ = false;
    transport_->DeInit();
}

void
ExtWdg::Run()
{
    int32_t res = transport_->Connect();
    if(res != 0) {
        EW_ERROR << "ExtWdg::Run() failed!";
        return;
    }
    EW_INFO << "ExtWdg::Run() success!";
}

void
ExtWdg::Stop()
{
    int32_t res = transport_->DisConnect();
    if(res != 0) {
        EW_ERROR << "ExtWdg::Stop() failed!";
        return;
    }
    EW_INFO << "ExtWdg::Stop() success!";
}

}  // namespace extwdg
}  // namespace netaos
}  // namespace hozon