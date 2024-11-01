// Copyright 2016 Proyectos y Sistemas de Mantenimiento SL (eProsima).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*!
 * @file HelloWorld.h
 * This header file contains the declaration of the described types in the IDL file.
 *
 * This file was generated by the tool fastddsgen.
 */
#ifndef _FAST_DDS_GENERATED_HELLOWORLD_H_
#define _FAST_DDS_GENERATED_HELLOWORLD_H_

#include <array>
#include <bitset>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

// #include <fastcdr/cdr/fixed_size_string.hpp>
// #include <fastcdr/xcdr/optional.hpp>
#include <fastdds/rtps/common/SerializedPayload.h>

#if defined(_WIN32)
#if defined(EPROSIMA_USER_DLL_EXPORT)
#define eProsima_user_DllExport __declspec(dllexport)
#else
#define eProsima_user_DllExport
#endif  // EPROSIMA_USER_DLL_EXPORT
#else
#define eProsima_user_DllExport
#endif  // _WIN32

#if defined(_WIN32)
#if defined(EPROSIMA_USER_DLL_EXPORT)
#if defined(HELLOWORLD_SOURCE)
#define HELLOWORLD_DllAPI __declspec(dllexport)
#else
#define HELLOWORLD_DllAPI __declspec(dllimport)
#endif  // HELLOWORLD_SOURCE
#else
#define HELLOWORLD_DllAPI
#endif  // EPROSIMA_USER_DLL_EXPORT
#else
#define HELLOWORLD_DllAPI
#endif  // _WIN32

/*!
 * @brief This class represents the structure BagDataType defined by the user in the IDL file.
 * @ingroup BagDataType
 */
class BagDataType {
   public:
    /*!
     * @brief Default constructor.
     */
    eProsima_user_DllExport BagDataType();

    /*!
     * @brief Default destructor.
     */
    eProsima_user_DllExport ~BagDataType();

    /*!
     * @brief Copy constructor.
     * @param x Reference to the object BagDataType that will be copied.
     */
    eProsima_user_DllExport BagDataType(const BagDataType& x);

    /*!
     * @brief Move constructor.
     * @param x Reference to the object BagDataType that will be copied.
     */
    eProsima_user_DllExport BagDataType(BagDataType&& x) noexcept;

    /*!
     * @brief Copy assignment.
     * @param x Reference to the object BagDataType that will be copied.
     */
    eProsima_user_DllExport BagDataType& operator=(const BagDataType& x);

    /*!
     * @brief Move assignment.
     * @param x Reference to the object BagDataType that will be copied.
     */
    eProsima_user_DllExport BagDataType& operator=(BagDataType&& x) noexcept;

    /*!
     * @brief Comparison operator.
     * @param x BagDataType object to compare.
     */
    eProsima_user_DllExport bool operator==(const BagDataType& x) const;

    /*!
     * @brief Comparison operator.
     * @param x BagDataType object to compare.
     */
    eProsima_user_DllExport bool operator!=(const BagDataType& x) const;

    std::shared_ptr<eprosima::fastrtps::rtps::SerializedPayload_t> m_payload;
};

#endif  // _FAST_DDS_GENERATED_HELLOWORLD_H_
