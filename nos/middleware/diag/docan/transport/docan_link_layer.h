/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanLinkLayer Header
 */


#ifndef DOCAN_LINKLAYER_H_
#define DOCAN_LINKLAYER_H_


#include <stdint.h>

typedef enum {
    STANDARD,
    EXTENDED
} Identifier_Type_t;
typedef struct {
    Identifier_Type_t Type;
    uint32_t Id;
} Identifier_t;
typedef uint8_t DLC_t;
typedef uint8_t* Data_t;
typedef enum {
    COMPLETE,
    NOT_COMPLETE,
    ABORTED
} Transfer_Status_t;

typedef struct {
    /* data */
    Identifier_t Identifier;
    Transfer_Status_t Transfer_Status;
} docan_link_confirm_t;

typedef struct {
    /* data */
    Identifier_t Identifier;
    DLC_t DLC;
    Data_t Data;
} docan_link_indicate_t;


namespace hozon {
namespace netaos {
namespace diag {

typedef int32_t (*docan_link_indication_callback_t)(Identifier_t, DLC_t, Data_t);
typedef int32_t (*docan_link_confirm_callback_t)(Identifier_t, Transfer_Status_t);

class DocanLinkLayer {
public:
    DocanLinkLayer(docan_link_indication_callback_t indication_callback,
        docan_link_confirm_callback_t confirm_callback);
    virtual ~DocanLinkLayer();

    int32_t Init();
    int32_t Start();
    int32_t Stop();

    int32_t L_Data_Request(Identifier_t Identifier, DLC_t DLC, Data_t Data);

private:
    DocanLinkLayer(const DocanLinkLayer &);
    DocanLinkLayer & operator = (const DocanLinkLayer &);

    int32_t L_Data_Confirm(Identifier_t Identifier, Transfer_Status_t Transfer_Status);
    int32_t L_Data_Indication(Identifier_t Identifier, DLC_t DLC, Data_t Data);

private:
    docan_link_indication_callback_t    indication_callback_;
    docan_link_confirm_callback_t       confirm_callback_;

};

} // end of diag
} // end of netaos
} // end of hozon
#endif  //
