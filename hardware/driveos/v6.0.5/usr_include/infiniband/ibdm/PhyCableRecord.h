/*
 * Copyright (c) 2004-2021 Mellanox Technologies LTD. All rights reserved.
 * Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This software is available to you under the terms of the
 * OpenIB.org BSD license included below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */


#ifndef IBDM_PHY_CABLE_H
#define IBDM_PHY_CABLE_H

#include <ostream>
#include <stdint.h>

using namespace std;


class PhyCableRecord
{
public:

   struct ModuleRecord
   {
        uint8_t ethernet_compliance_code;
        uint8_t ext_ethernet_compliance_code;
        uint8_t cable_breakout;
        uint8_t cable_technology;
        uint8_t cable_power_class;
        uint8_t cable_identifier;
        uint8_t cable_length;
        uint8_t cable_vendor;
        uint8_t cable_type;
        uint8_t cable_tx_equalization;
        uint8_t cable_rx_emphasis;
        uint8_t cable_rx_amp;
        uint8_t max_power;
        uint8_t cable_attenuation_5g;
        uint8_t cable_attenuation_7g;
        uint8_t cable_attenuation_12g;
        uint8_t cable_attenuation_25g;
        uint8_t tx_cdr_state;
        uint8_t rx_cdr_state;
        uint8_t tx_cdr_cap;
        uint8_t rx_cdr_cap;
        uint8_t cable_rx_post_emphasis;
        char vendor_name[17];
        char vendor_pn[17];
        char vendor_rev[5];
        uint32_t fw_version;
        char vendor_sn[17];
        uint16_t voltage;
        uint16_t temperature;
        uint16_t rx_power_lane1;
        uint16_t rx_power_lane0;
        uint16_t rx_power_lane3;
        uint16_t rx_power_lane2;
        uint16_t rx_power_lane5;
        uint16_t rx_power_lane4;
        uint16_t rx_power_lane7;
        uint16_t rx_power_lane6;
        uint16_t tx_power_lane1;
        uint16_t tx_power_lane0;
        uint16_t tx_power_lane3;
        uint16_t tx_power_lane2;
        uint16_t tx_power_lane5;
        uint16_t tx_power_lane4;
        uint16_t tx_power_lane7;
        uint16_t tx_power_lane6;
        uint16_t tx_bias_lane1;
        uint16_t tx_bias_lane0;
        uint16_t tx_bias_lane3;
        uint16_t tx_bias_lane2;
        uint16_t tx_bias_lane5;
        uint16_t tx_bias_lane4;
        uint16_t tx_bias_lane7;
        uint16_t tx_bias_lane6;
        uint16_t temperature_low_th;
        uint16_t temperature_high_th;
        uint16_t voltage_low_th;
        uint16_t voltage_high_th;
        uint16_t rx_power_low_th;
        uint16_t rx_power_high_th;
        uint16_t tx_power_low_th;
        uint16_t tx_power_high_th;
        uint16_t tx_bias_low_th;
        uint16_t tx_bias_high_th;
        uint16_t wavelength;
        uint16_t smf_length;
        uint8_t did_cap;
        uint8_t rx_power_type;
        uint8_t module_st;
        uint8_t ib_compliance_code;
        uint8_t active_set_media_compliance_code;
        uint8_t active_set_host_compliance_code;
        uint8_t ib_width;
        uint8_t monitor_cap_mask;
        uint8_t nbr_100;
        uint8_t nbr_250;
        uint8_t dp_st_lane7;
        uint8_t dp_st_lane6;
        uint8_t dp_st_lane5;
        uint8_t dp_st_lane4;
        uint8_t dp_st_lane3;
        uint8_t dp_st_lane2;
        uint8_t dp_st_lane1;
        uint8_t dp_st_lane0;
        uint8_t length_om5;
        uint8_t length_om4;
        uint8_t length_om3;
        uint8_t length_om2;
        uint8_t memory_map_rev;
        uint16_t wavelength_tolerance;
        uint8_t length_om1;
        uint32_t memory_map_compliance;
        uint64_t date_code;
        uint32_t vendor_oui;
        uint8_t connector_type;
        uint8_t rx_output_valid;
        uint8_t rx_input_valid;
        uint8_t tx_input_freq_sync;
        uint8_t error_code;

        private:
        enum CableModuleInfoIdentifier {
                CABLE_MODULE_INFO_IDENTIFIER_QSFP28             = 0,
                CABLE_MODULE_INFO_IDENTIFIER_QSFP_PLUS,
                CABLE_MODULE_INFO_IDENTIFIER_SFP28_SFP_plus,
                CABLE_MODULE_INFO_IDENTIFIER_QSA,
                CABLE_MODULE_INFO_IDENTIFIER_Backplane,
                CABLE_MODULE_INFO_IDENTIFIER_SFP_DD,
                CABLE_MODULE_INFO_IDENTIFIER_QSFP_DD,
                CABLE_MODULE_INFO_IDENTIFIER_QSFP_CMIS,
                CABLE_MODULE_INFO_IDENTIFIER_OSFP,
                CABLE_MODULE_INFO_IDENTIFIER_C2C,
                CABLE_MODULE_INFO_IDENTIFIER_DSFP,
                CABLE_MODULE_INFO_IDENTIFIER_QSFP_SPLIT_CABLE
        };


        public:
        string ConvertRevisionToStr() const;
        string ConvertCableIdentifierToStr() const;
        string ConvertCableLengthToStr(bool is_csv, const string &na_val) const;
        bool IsCMISCable() const;
        string ConvertCableLengthSMFiberToStr() const;
        string ConvertCableLengthOMXToStr(u_int8_t om_index) const;
        u_int8_t SelectTransmitterTechnology() const;
        string ConvertTransmitterTechnologyToStr(const string &na_val) const;
        string ConvertIBComplianceCodeToStr(const string &na_val) const;
        string ConvertTemperatureToStr(u_int16_t temp, bool is_csv) const;
        string ConvertVoltageToStr(u_int16_t vcc, bool is_csv) const;
        u_int8_t SelectNominalBR() const;
        string ConvertCDREnableTxRxToStr(bool is_rx, const string &na_val, bool is_csv = false) const;
        bool IsModule() const;
        bool IsActiveCable() const;
        bool IsPassiveCable() const;
        string ConvertTxEQRxAMPRxEMPToStr(u_int8_t val, bool is_csv, const string &na_val) const;
        string ConvertFWVersionToStr(const string& na_val) const;
        string ConvertAttenuationToStr(bool is_csv) const;
        string ConvertDateCodeToStr(const string &na_val) const;
        string ConvertMaxPowerToStr(const string &na_val) const;
   };

   struct LatchedRecord
   {
        uint8_t dp_fw_fault;
        uint8_t mod_fw_fault;
        uint8_t vcc_flags;
        uint8_t temp_flags;
        uint8_t tx_ad_eq_fault;
        uint8_t tx_cdr_lol;
        uint8_t tx_los;
        uint8_t tx_fault;
        uint8_t tx_power_lo_war;
        uint8_t tx_power_hi_war;
        uint8_t tx_power_lo_al;
        uint8_t tx_power_hi_al;
        uint8_t tx_bias_lo_war;
        uint8_t tx_bias_hi_war;
        uint8_t tx_bias_lo_al;
        uint8_t tx_bias_hi_al;
        uint8_t rx_cdr_lol;
        uint8_t rx_los;
        uint8_t rx_power_lo_war;
        uint8_t rx_power_hi_war;
        uint8_t rx_power_lo_al;
        uint8_t rx_power_hi_al;
        uint8_t rx_output_valid_change;
        uint8_t rx_input_valid_change;
   };

public:
    //todo Do Not allow copy
    PhyCableRecord(const string & source, ModuleRecord* p_inMod = NULL, LatchedRecord* p_inLat = NULL):
        m_source(source), p_module(p_inMod), p_latched(p_inLat) {}
    ~PhyCableRecord() {
        delete p_module;
        delete p_latched;
    }

    void ToCSVStream(ostream &stream) const;
    void ToFileStream(ostream &stream) const;
    int GetTemperatureAlarms() const;
    int GetTemperatureErrorsByTreshold() const;
    string GetTemperatureStr() const;
    string GetHighTemperatureThresholdStr() const;
    string GetLowTemperatureThresholdStr() const;

    string         m_source;
    ModuleRecord  *p_module;
    LatchedRecord *p_latched;

private:
    string VendorOUIToStr() const;
    string CableIdentifierToStr(bool is_csv = false) const;
    string CablePowerClassToStr(bool is_csv = false) const;
    string BitrateToStr(bool is_csv = false) const;
    string DescToCsvDesc(const string &desc) const;
    string VendorToStr() const;
    string VendorPnToStr() const;
    string VendorSnToStr() const;
    string RevisionToStr() const;
    string CableLengthToStr(bool is_csv = false) const;
    string TypeToStr(bool is_csv = false) const;
    string SupportedSpeedToStr(bool is_csv = false) const;
    string CDREnableTxRXToStr() const;
    string InputEqToStr(bool is_csv = false) const;
    string OutputAmpToStr(bool is_csv = false) const;
    string FWVersionToStr(bool is_csv = false) const;
    string AttenuationToStr(bool is_csv = false) const;
    string RXPowerTypeToStr() const;
    double dBm_to_mW(double mW) const;
    string PowerLineToStr(double lane, bool is_csv) const;
    string EmptyPowerLineToStr(bool is_csv) const;
    string RX1PowerToStr(bool is_csv = false) const;
    string RX2PowerToStr(bool is_csv = false) const;
    string RX3PowerToStr(bool is_csv = false) const;
    string RX4PowerToStr(bool is_csv = false) const;
    string TXBiasToStr(bool is_csv, double bias_line) const;
    string TXBias1ToStr(bool is_csv = false) const;
    string TXBias2ToStr(bool is_csv = false) const;
    string TXBias3ToStr(bool is_csv = false) const;
    string TXBias4ToStr(bool is_csv = false) const;
    string TX1PowerToStr(bool is_csv = false) const;
    string TX2PowerToStr(bool is_csv = false) const;
    string TX3PowerToStr(bool is_csv = false) const;
    string TX4PowerToStr(bool is_csv = false) const;
    string TechnologyToStr(bool is_csv = false) const;
    string ComplanceCodeToStr(bool is_csv = false) const;
    string DateCodeToStr(bool is_csv = false) const;
    string LotToStr(bool is_csv = false) const;
    string LengthSMFiberToStr() const;
    string LengthOmxToStr(u_int8_t index) const;
    string MaxPowerToStr(bool is_csv) const;

    //Latched part
    string LatchedTxRxIndicatorToStr() const;
    string LatchedTxRxLolIndicatorToStr() const;
    string LatchedAdaptiveEqualizationFaultToStr() const;
    string LatchedTempAlarmAndWarningToStr() const;
    string LatchedVoltageAlarmAndWarningToStr() const;
    uint16_t QuadroToInt(uint8_t hi_al, uint8_t lo_al,
                    uint8_t hi_pw, uint8_t lo_pw) const;
    string RXPowerAlarmAndWarningToStr() const;
    string TXBiasAlarmAndWarningToStr() const;
    string TXPowerAlarmAndWarningToStr() const;
    string OuputPreEmpToStr(bool is_csv = false) const;
    string OuputPostEmpToStr(bool is_csv = false) const;

};

#endif
