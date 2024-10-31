/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanLinkLayer implement
 */

#include "docan_link_layer.h"

namespace hozon {
namespace netaos {
namespace diag {


DocanLinkLayer::DocanLinkLayer(docan_link_indication_callback_t indication_callback,
        docan_link_confirm_callback_t confirm_callback)
    : indication_callback_(indication_callback)
    , confirm_callback_(confirm_callback)
{

}

DocanLinkLayer::~DocanLinkLayer()
{
}

int32_t
DocanLinkLayer::Init()
{
    return 0;
}

int32_t
DocanLinkLayer::Start()
{
    return 0;
}

int32_t
DocanLinkLayer::Stop()
{
    return 0;
}

int32_t
DocanLinkLayer::L_Data_Request(Identifier_t Identifier, DLC_t DLC, Data_t Data)
{
    // send to can ecu.
    return 0;
}

int32_t
DocanLinkLayer::L_Data_Confirm(Identifier_t Identifier, Transfer_Status_t Transfer_Status)
{
    int32_t ret = -1;
    if (nullptr != confirm_callback_) {
        ret = confirm_callback_(Identifier, Transfer_Status);
    }
    return ret;
}

int32_t
DocanLinkLayer::L_Data_Indication(Identifier_t Identifier, DLC_t DLC, Data_t Data)
{
    int32_t ret = -1;
    if (nullptr != indication_callback_) {
        ret = indication_callback_(Identifier, DLC, Data);
    }
    return ret;
}

} // end of diag
} // end of netaos
} // end of hozon
/* EOF */