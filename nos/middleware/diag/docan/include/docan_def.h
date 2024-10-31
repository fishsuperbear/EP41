/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  DoCan defination Header
 */

#ifndef DOCAN_DEF_H_
#define DOCAN_DEF_H_

namespace hozon {
namespace netaos {
namespace diag {




typedef enum {
    OK  = 1,
    ERROR = 0xFF +2,
    TIMEOUT_A,
    TIMEOUT_Bs,
    TIMEOUT_Cr,
    WRONG_SN,
    INVALID_FS,
    UNEXP_PDU,
    WFT_OVRN,
    BUFFER_OVFLW,
    TIMEOUT_P2StarServer,
    RX_ON,
    WRONG_PARAMETER,
    WRONG_VALUE,
    USER_CANCEL,
} docan_result_t;

} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_DEF_H_
/* EOF */
