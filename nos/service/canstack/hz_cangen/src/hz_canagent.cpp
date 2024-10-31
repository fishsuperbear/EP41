#include "hz_canagent.h"

namespace hozon {
namespace netaos {
namespace canstack {
namespace cangen {

CanAgent::CanAgent() {
    can_pack_info_ = std::make_shared<CanPackInfo_t>();
}

// void CanAgent::Init() {}
CanAgent::~CanAgent() {}

void CanAgent::PackFrame102(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_102_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_102_pack(dst_p, src_p, 64);
}

void CanAgent::PackFrame120(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_120_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_120_pack(dst_p, src_p, 64);
}

void CanAgent::PackFrame180(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_180_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_180_pack(dst_p, src_p, 64);
}

void CanAgent::PackFrame181(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_181_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_181_pack(dst_p, src_p, 64);
}

void CanAgent::PackFrame1EC(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_1_ec_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_1_ec_pack(dst_p, src_p, 64);
}

void CanAgent::PackFrame1ED(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_1_ed_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_1_ed_pack(dst_p, src_p, 64);
}

void CanAgent::PackFrame211(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_211_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_211_pack(dst_p, src_p, 64);
}

void CanAgent::PackFrame230(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_230_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_230_pack(dst_p, src_p, 64);
}

void CanAgent::PackFrame231(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_231_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_231_pack(dst_p, src_p, 64);
}

void CanAgent::PackFrame232(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_232_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_232_pack(dst_p, src_p, 64);
}

void CanAgent::PackFrame233(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_233_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_233_pack(dst_p, src_p, 64);
}

void CanAgent::PackFrame234(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_234_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_234_pack(dst_p, src_p, 64);
}

void CanAgent::PackFrame264(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_264_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_264_pack(dst_p, src_p, 64);
}

void CanAgent::PackFrame135(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_event_135_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_event_135_pack(dst_p, src_p, 64);
}

void CanAgent::PackFrame100(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_100_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_100_pack(dst_p, src_p, 64);
};

void CanAgent::PackFrame112(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_112_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_112_pack(dst_p, src_p, 64);
};

void CanAgent::PackFrame1A2(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_1_a2_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_1_a2_pack(dst_p, src_p, 64);
};

void CanAgent::PackFrame200(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_200_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_200_pack(dst_p, src_p, 64);
};

void CanAgent::PackFrame201(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_201_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_201_pack(dst_p, src_p, 64);
};

void CanAgent::PackFrameA2(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_a2_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_a2_pack(dst_p, src_p, 64);
};

void CanAgent::PackFrameE3(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_e3_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_e3_pack(dst_p, src_p, 64);
};

void CanAgent::PackFrameE5(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_e5_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_cyc_e5_pack(dst_p, src_p, 64);
};

void CanAgent::PackFrame127(uint8_t* dst_p, const struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_event_0x127_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_event_0x127_pack(dst_p, src_p, 64);
};

void CanAgent::PackFrame1E3(uint8_t* dst_p, struct ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_event_0x1_e3_t* src_p) {
    ep40_canfd_mcu_soc_v4_1_adcs_mcu_soc_tx_event_0x1_e3_pack(dst_p, src_p, 64);
};
}  // namespace cangen
}  // namespace canstack
}  // namespace netaos
}  // namespace hozon
