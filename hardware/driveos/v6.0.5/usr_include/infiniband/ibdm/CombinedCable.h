/*
 * Copyright (c) 2004-2021 Mellanox Technologies LTD. All rights reserved.
 * Copyright (c) 2021-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#ifndef IBDM_COMBINED_CABLE_H
#define IBDM_COMBINED_CABLE_H

#include <ostream>

#include "infiniband/ibdm/cable/PhyCableRecord.h"
#include "infiniband/ibdm/cable/CableRecord.h"

using namespace std;

#define CABLE_HIGH_TEMP_MASK 0x01
#define CABLE_LOW_TEMP_MASK  0x02

class CombinedCableInfo {
public:
    CombinedCableInfo() = delete;
    CombinedCableInfo(const CableRecord &cableRec): p_phy(NULL) {
        p_cable = new CableRecord(cableRec);
    }
    CombinedCableInfo(const string & source, PhyCableRecord::ModuleRecord *p_inMod,
                    PhyCableRecord::LatchedRecord *pInLat): p_cable(NULL) {
        p_phy = new PhyCableRecord(source, p_inMod, pInLat);
    }
    ~CombinedCableInfo();
    void ToCSVStream(ostream &stream) const;
    void ToFileStream(ostream &stream) const;
    int GetTemperatureAlarms() const;
    int GetTemperatureErrorsByTreshold() const;
    string GetTemperatureStr() const;
    string GetHighTemperatureThresholdStr() const;
    string GetLowTemperatureThresholdStr() const;

    static string GetCSVHeader();
    static string TemperatureToStr(u_int8_t cableType, int8_t temp, const string& defaultVal);
    static string VoltageToStr(u_int16_t vcc);
    static string SupportedSpeedToStr(u_int8_t supportedSpeed, const string& defaultVal);
    static string CableTypeToStr(u_int8_t type, const string& defaultVal);

private:

    CableRecord    *p_cable;
    PhyCableRecord *p_phy;
};

#endif
