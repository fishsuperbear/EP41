#ifndef SM_ALG_STRUCT_PARKINGLOT_INFO_H__
#define SM_ALG_STRUCT_PARKINGLOT_INFO_H__

#include <stdint.h>
#include <vector>


enum ParkStatus {
    ParkStatus_FREE = 0,
    ParkStatus_OCCUPIED = 1,
    ParkStatus_UNKOWN = 2
};

/*******************************************************************************
    结构 名        :  AlgParkingLotOut
    功能描述       :  AlgParkingLotOut
*******************************************************************************/
struct AlgParkingLotOut {
    int32_t parkingSeq;  // id
    ParkStatus status;
};

/*******************************************************************************
    结构 名        :  AlgParkingLotOutArray
    功能描述       :  停车位检测结果列表
*******************************************************************************/
struct AlgParkingLotOutArray {
    uint32_t optParkingSeq{};
    std::vector<AlgParkingLotOut> parkingLots{};
};

#endif  //  SM_ALG_STRUCT_PARKINGLOT_INFO_H__
