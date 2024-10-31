/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STATEMACHINE_IMPL_TYPE_WORKINGSTATUS_H
#define HOZON_STATEMACHINE_IMPL_TYPE_WORKINGSTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint8_t.h"

namespace hozon {
namespace statemachine {
struct WorkingStatus {
    ::UInt8 processing_status;
    ::UInt8 error_code;
    ::UInt8 perception_warninginfo;
    ::UInt8 perception_ADCS4_Tex;
    ::UInt8 perception_ADCS4_PA_failinfo;
    ::UInt8 TBA_Distance;
    ::UInt8 TBA;
    ::UInt8 TBA_text;
    ::uint8_t HPA;
    ::uint8_t HPA_PathOnParkArea;
    ::uint8_t HPA_PathStoreSts;
    ::uint8_t HPA_learnpathStSw;
    ::uint8_t HPA_PathlearnSts;
    ::uint8_t HPA_PathlearningWorkSts;
    ::uint8_t HPA_PointInParkslot;
    ::uint8_t HPA_PathwaytoCloudWorkSts;
    ::uint8_t HPA_GuideSts;
    ::uint8_t HPA_ReturnButton;
    ::uint8_t HPA_PathexistSts;
    ::uint8_t HPA_distance;
    ::uint8_t HPA_Pathavailable_ID;
    ::uint8_t HPA_CrossingNumber;
    ::uint8_t perception_ADCS4_HPA_failinfo;
    ::uint8_t HPA_LocalizationSts;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(processing_status);
        fun(error_code);
        fun(perception_warninginfo);
        fun(perception_ADCS4_Tex);
        fun(perception_ADCS4_PA_failinfo);
        fun(TBA_Distance);
        fun(TBA);
        fun(TBA_text);
        fun(HPA);
        fun(HPA_PathOnParkArea);
        fun(HPA_PathStoreSts);
        fun(HPA_learnpathStSw);
        fun(HPA_PathlearnSts);
        fun(HPA_PathlearningWorkSts);
        fun(HPA_PointInParkslot);
        fun(HPA_PathwaytoCloudWorkSts);
        fun(HPA_GuideSts);
        fun(HPA_ReturnButton);
        fun(HPA_PathexistSts);
        fun(HPA_distance);
        fun(HPA_Pathavailable_ID);
        fun(HPA_CrossingNumber);
        fun(perception_ADCS4_HPA_failinfo);
        fun(HPA_LocalizationSts);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(processing_status);
        fun(error_code);
        fun(perception_warninginfo);
        fun(perception_ADCS4_Tex);
        fun(perception_ADCS4_PA_failinfo);
        fun(TBA_Distance);
        fun(TBA);
        fun(TBA_text);
        fun(HPA);
        fun(HPA_PathOnParkArea);
        fun(HPA_PathStoreSts);
        fun(HPA_learnpathStSw);
        fun(HPA_PathlearnSts);
        fun(HPA_PathlearningWorkSts);
        fun(HPA_PointInParkslot);
        fun(HPA_PathwaytoCloudWorkSts);
        fun(HPA_GuideSts);
        fun(HPA_ReturnButton);
        fun(HPA_PathexistSts);
        fun(HPA_distance);
        fun(HPA_Pathavailable_ID);
        fun(HPA_CrossingNumber);
        fun(perception_ADCS4_HPA_failinfo);
        fun(HPA_LocalizationSts);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("processing_status", processing_status);
        fun("error_code", error_code);
        fun("perception_warninginfo", perception_warninginfo);
        fun("perception_ADCS4_Tex", perception_ADCS4_Tex);
        fun("perception_ADCS4_PA_failinfo", perception_ADCS4_PA_failinfo);
        fun("TBA_Distance", TBA_Distance);
        fun("TBA", TBA);
        fun("TBA_text", TBA_text);
        fun("HPA", HPA);
        fun("HPA_PathOnParkArea", HPA_PathOnParkArea);
        fun("HPA_PathStoreSts", HPA_PathStoreSts);
        fun("HPA_learnpathStSw", HPA_learnpathStSw);
        fun("HPA_PathlearnSts", HPA_PathlearnSts);
        fun("HPA_PathlearningWorkSts", HPA_PathlearningWorkSts);
        fun("HPA_PointInParkslot", HPA_PointInParkslot);
        fun("HPA_PathwaytoCloudWorkSts", HPA_PathwaytoCloudWorkSts);
        fun("HPA_GuideSts", HPA_GuideSts);
        fun("HPA_ReturnButton", HPA_ReturnButton);
        fun("HPA_PathexistSts", HPA_PathexistSts);
        fun("HPA_distance", HPA_distance);
        fun("HPA_Pathavailable_ID", HPA_Pathavailable_ID);
        fun("HPA_CrossingNumber", HPA_CrossingNumber);
        fun("perception_ADCS4_HPA_failinfo", perception_ADCS4_HPA_failinfo);
        fun("HPA_LocalizationSts", HPA_LocalizationSts);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("processing_status", processing_status);
        fun("error_code", error_code);
        fun("perception_warninginfo", perception_warninginfo);
        fun("perception_ADCS4_Tex", perception_ADCS4_Tex);
        fun("perception_ADCS4_PA_failinfo", perception_ADCS4_PA_failinfo);
        fun("TBA_Distance", TBA_Distance);
        fun("TBA", TBA);
        fun("TBA_text", TBA_text);
        fun("HPA", HPA);
        fun("HPA_PathOnParkArea", HPA_PathOnParkArea);
        fun("HPA_PathStoreSts", HPA_PathStoreSts);
        fun("HPA_learnpathStSw", HPA_learnpathStSw);
        fun("HPA_PathlearnSts", HPA_PathlearnSts);
        fun("HPA_PathlearningWorkSts", HPA_PathlearningWorkSts);
        fun("HPA_PointInParkslot", HPA_PointInParkslot);
        fun("HPA_PathwaytoCloudWorkSts", HPA_PathwaytoCloudWorkSts);
        fun("HPA_GuideSts", HPA_GuideSts);
        fun("HPA_ReturnButton", HPA_ReturnButton);
        fun("HPA_PathexistSts", HPA_PathexistSts);
        fun("HPA_distance", HPA_distance);
        fun("HPA_Pathavailable_ID", HPA_Pathavailable_ID);
        fun("HPA_CrossingNumber", HPA_CrossingNumber);
        fun("perception_ADCS4_HPA_failinfo", perception_ADCS4_HPA_failinfo);
        fun("HPA_LocalizationSts", HPA_LocalizationSts);
    }

    bool operator==(const ::hozon::statemachine::WorkingStatus& t) const
    {
        return (processing_status == t.processing_status) && (error_code == t.error_code) && (perception_warninginfo == t.perception_warninginfo) && (perception_ADCS4_Tex == t.perception_ADCS4_Tex) && (perception_ADCS4_PA_failinfo == t.perception_ADCS4_PA_failinfo) && (TBA_Distance == t.TBA_Distance) && (TBA == t.TBA) && (TBA_text == t.TBA_text) && (HPA == t.HPA) && (HPA_PathOnParkArea == t.HPA_PathOnParkArea) && (HPA_PathStoreSts == t.HPA_PathStoreSts) && (HPA_learnpathStSw == t.HPA_learnpathStSw) && (HPA_PathlearnSts == t.HPA_PathlearnSts) && (HPA_PathlearningWorkSts == t.HPA_PathlearningWorkSts) && (HPA_PointInParkslot == t.HPA_PointInParkslot) && (HPA_PathwaytoCloudWorkSts == t.HPA_PathwaytoCloudWorkSts) && (HPA_GuideSts == t.HPA_GuideSts) && (HPA_ReturnButton == t.HPA_ReturnButton) && (HPA_PathexistSts == t.HPA_PathexistSts) && (HPA_distance == t.HPA_distance) && (HPA_Pathavailable_ID == t.HPA_Pathavailable_ID) && (HPA_CrossingNumber == t.HPA_CrossingNumber) && (perception_ADCS4_HPA_failinfo == t.perception_ADCS4_HPA_failinfo) && (HPA_LocalizationSts == t.HPA_LocalizationSts);
    }
};
} // namespace statemachine
} // namespace hozon


#endif // HOZON_STATEMACHINE_IMPL_TYPE_WORKINGSTATUS_H
