#pragma once

#include <stdint.h>

#include <functional>
#include <map>
#include <memory>
#include <mutex>

#include "ep40_canfd_mcu_soc_v4_1_.h"

namespace hozon {
namespace netaos {
namespace canstack {
namespace cangen {

class CanAgent {
   public:
    static CanAgent& Instance() {
        static CanAgent instance;
        return instance;
    }

    ~CanAgent();

    void PackFrame102(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_102_t* src_p);
    void PackFrame120(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_120_t* src_p);
    void PackFrame180(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_180_t* src_p);
    void PackFrame181(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_181_t* src_p);
    void PackFrame1EC(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_1_ec_t* src_p);
    void PackFrame1ED(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_1_ed_t* src_p);
    void PackFrame211(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_211_t* src_p);
    void PackFrame230(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_230_t* src_p);
    void PackFrame231(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_231_t* src_p);
    void PackFrame232(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_232_t* src_p);
    void PackFrame233(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_233_t* src_p);
    void PackFrame234(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_234_t* src_p);
    void PackFrame264(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_264_t* src_p);
    void PackFrame135(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_event_135_t* src_p);

    void PackFrame100(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_100_t* src_p);
    void PackFrame112(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_112_t* src_p);
    void PackFrame1A2(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_1_a2_t* src_p);
    void PackFrame200(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_200_t* src_p);
    void PackFrame201(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_201_t* src_p);
    void PackFrameA2(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_a2_t* src_p);
    void PackFrameE3(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_e3_t* src_p);
    void PackFrameE5(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_e5_t* src_p);
    void PackFrame127(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_event_0x127_t* src_p);
    void PackFrame1E3(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_event_0x1_e3_t* src_p);

    typedef struct CanPackInfo {
        ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_102_t can_frame_102_;
        ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_120_t can_frame_120_;
        ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_180_t can_frame_180_;
        ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_181_t can_frame_181_;
        ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_1_ec_t can_frame_1_ec_;
        ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_1_ed_t can_frame_1_ed_;
        ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_211_t can_frame_211_;
        ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_230_t can_frame_230_;
        ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_231_t can_frame_231_;
        ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_232_t can_frame_232_;
        ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_233_t can_frame_233_;
        ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_234_t can_frame_234_;
        ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_264_t can_frame_264_;
        ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_event_135_t can_frame_135_;

        ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_100_t can_frame_100_;
        ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_112_t can_frame_112_;
        ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_1_a2_t can_frame_1a2_;
        ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_200_t can_frame_200_;
        ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_201_t can_frame_201_;
        ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_a2_t can_frame_a2_;
        ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_e3_t can_frame_e3_;
        ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_e5_t can_frame_e5_;
        ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_event_0x127_t can_frame_127_;
        ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_event_0x1_e3_t can_frame_1e3_;
    } CanPackInfo_t;

    std::shared_ptr<CanPackInfo_t> can_pack_info_;

   private:
    CanAgent();

    // static std::mutex mtx_;
    // static CanAgent instance;
};

}  // namespace cangen
}  // namespace canstack
}  // namespace netaos
}  // namespace hozon
