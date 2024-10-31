/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_ALGIGST_H
#define HOZON_CHASSIS_IMPL_TYPE_ALGIGST_H
#include <cfloat>
#include <cmath>

namespace hozon {
namespace chassis {
struct AlgIgSt {
    bool IG_OFF;
    bool ACC;
    bool IG_ON;
    bool Start;
    bool Remote_IG_ON;
    bool reserve_1;
    bool reserve_2;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(IG_OFF);
        fun(ACC);
        fun(IG_ON);
        fun(Start);
        fun(Remote_IG_ON);
        fun(reserve_1);
        fun(reserve_2);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(IG_OFF);
        fun(ACC);
        fun(IG_ON);
        fun(Start);
        fun(Remote_IG_ON);
        fun(reserve_1);
        fun(reserve_2);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("IG_OFF", IG_OFF);
        fun("ACC", ACC);
        fun("IG_ON", IG_ON);
        fun("Start", Start);
        fun("Remote_IG_ON", Remote_IG_ON);
        fun("reserve_1", reserve_1);
        fun("reserve_2", reserve_2);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("IG_OFF", IG_OFF);
        fun("ACC", ACC);
        fun("IG_ON", IG_ON);
        fun("Start", Start);
        fun("Remote_IG_ON", Remote_IG_ON);
        fun("reserve_1", reserve_1);
        fun("reserve_2", reserve_2);
    }

    bool operator==(const ::hozon::chassis::AlgIgSt& t) const
    {
        return (IG_OFF == t.IG_OFF) && (ACC == t.ACC) && (IG_ON == t.IG_ON) && (Start == t.Start) && (Remote_IG_ON == t.Remote_IG_ON) && (reserve_1 == t.reserve_1) && (reserve_2 == t.reserve_2);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_ALGIGST_H
