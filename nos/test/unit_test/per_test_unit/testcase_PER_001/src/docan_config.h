#ifndef __DOCAN_DATA_H__
#define __DOCAN_DATA_H__

#include <stdint.h>

#include <map>
#include <string>
#include <vector>

#include "struct2x/struct2x.h"  // SERIALIZE

struct N_EcuInfo {
    N_EcuInfo() {}
    std::string Name;
    std::string IfName;
    std::string CanType;   // 0: invalid, 1: can, 2: canfd
    std::string DiagType;  // 0: invalid, 1: local diag, 2: remote diag
    std::string TaType;    // 0: invalid, 1: physical addr, 2: functional addr
    std::string LogicalAddr;
    std::string CanidTx;
    std::string CanidRx;
    std::string FunctionAddr;
    uint8_t BS;
    uint8_t STmin;
    uint16_t WFTmax;
    uint16_t As;
    uint16_t Ar;
    uint16_t Bs;
    uint16_t Br;
    uint16_t Cs;
    uint16_t Cr;
    template <typename T>
    void serialize(T& t) {
        SERIALIZE(t, Name, IfName, CanType, DiagType, TaType, LogicalAddr, CanidTx, CanidRx, FunctionAddr, BS, STmin, WFTmax, As, Ar, Bs, Br, Cs, Cr);
        //     t.convert("a", Name)
        //         .convert("b", IfName)
        //         .convert("c", CanType)
        //         .convert("d", DiagType)
        //         .convert("e", TaType)
        //         .convert("f", LogicalAddr)
        //         .convert("g", CanidTx)
        //         .convert("h", CanidRx)
        //         .convert("i", FunctionAddr)
        //         .convert("g", BS)
        //         .convert("k", STmin)
        //         .convert("l", WFTmax)
        //         .convert("m", As)
        //         .convert("n", Ar)
        //         .convert("o", Bs)
        //         .convert("p", Br)
        //         .convert("q", Cs)
        //         .convert("r", Cr);
    }
};
struct N_EcuInfoList {
    N_EcuInfoList() {}
    std::vector<N_EcuInfo> DocanNodesConfiguration;
    template <typename T>
    void serialize(T& t) {
        SERIALIZE(t, DocanNodesConfiguration);
    }
};

#endif
