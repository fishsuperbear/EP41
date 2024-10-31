// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef ICASCADEDPROVIDER_HPP
#define ICASCADEDPROVIDER_HPP

#include <vector>
#include "NvSIPLCommon.hpp"
#include "nvscibuf.h"

class ICascadedProvider
{
public:
    virtual SIPLStatus GetNvSciBufAttrList( NvSciBufAttrList attrList ) = 0;
    virtual SIPLStatus RegisterNvSciBufObjs( const vector<NvSciBufObj>& outputNvSciBufOjbs ) = 0;
};

#endif // ICASCADEDPROVIDER_HPP