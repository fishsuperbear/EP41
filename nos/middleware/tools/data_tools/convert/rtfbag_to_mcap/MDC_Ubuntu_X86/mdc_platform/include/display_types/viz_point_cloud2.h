/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：PointCloud2
 */

#ifndef VIZ_POINT_CLOUD2_H
#define VIZ_POINT_CLOUD2_H

#include <cstdint>
#include "ara/core/vector.h"
#include "ara/core/string.h"
#include "viz_header.h"

namespace mdc {
namespace visual {
enum class DataType : uint8_t {
    DATATYPE_INT8 = 1U,
    DATATYPE_UINT8,
    DATATYPE_INT16,
    DATATYPE_UINT16,
    DATATYPE_INT32,
    DATATYPE_UINT32,
    DATATYPE_FLOAT32,
    DATATYPE_FLOAT64
};

enum class RetCode : int8_t {
    VIZ_OK = 0,
    SIZE_ERROR = -1,
    MEMCPY_COPY_ERROR = -2
};

struct PointField {
    ara::core::String name;
    uint32_t offset;
    DataType datatype;
    uint32_t count;
    PointField() : name(), offset(0U), datatype(DataType::DATATYPE_FLOAT32), count(0U) {}
    PointField(const ara::core::String& name_, const uint32_t& offset_,
    const DataType& dataType_, const uint32_t& count_)
        : name(name_), offset(offset_), datatype(dataType_), count(count_)
    {}
};

struct PointCloud2 {
    Header header;
    uint32_t height;
    uint32_t width;
    ara::core::Vector<PointField> fields;
    bool isBigendian;
    uint32_t pointStep;
    uint32_t rowStep;
    ara::core::Vector<uint8_t> data;
    bool isDense;
    PointCloud2()
        : header(),
          height(0U),
          width(0U),
          fields(),
          isBigendian(false),
          pointStep(0U),
          rowStep(0U),
          data(),
          isDense(false)
    {}
    PointCloud2(const Header& vHeader, const uint32_t& vHeight, const uint32_t& vWidth,
    const ara::core::Vector<PointField>& vFields, const bool& vIsBigendian, const uint32_t& vPointStep,
    const uint32_t& vRowStep, const ara::core::Vector<uint8_t>& vData, const bool& vIsDense)
        : header(vHeader),
          height(vHeight),
          width(vWidth),
          fields(vFields),
          isBigendian(vIsBigendian),
          pointStep(vPointStep),
          rowStep(vRowStep),
          data(vData),
          isDense(vIsDense)
    {}
};
struct PointXYZ {
    float x;
    float y;
    float z;
    PointXYZ() : x(0.0F), y(0.0F), z(0.0F) {}
    PointXYZ(const float& x_, const float& y_, const float& z_) : x(x_), y(y_), z(z_) {}
};
struct PointXYZI : public PointXYZ {
    float intensity;
    PointXYZI() : PointXYZ(), intensity(0.0F) {}
    PointXYZI(const float& x_, const float& y_, const float& z_, const float& intensity_)
        : PointXYZ(x_, y_, z_), intensity(intensity_) {}
};
struct PointXYZIR : public PointXYZI {
    uint32_t ring;
    PointXYZIR() : PointXYZI(), ring(0U) {}
    PointXYZIR(const float& x_, const float& y_, const float& z_, const float& intensity_, const uint32_t& ring_)
        : PointXYZI(x_, y_, z_, intensity_), ring(ring_)
    {}
};
struct PointXYZRGB : public PointXYZ {
    /* RGB color
     * The RGBA information is available either as separate r, g, b, or as a
     * packed uint32_t rgba value. To pack it, use:
     *
     * int rgb = ((int)r) << 16 | ((int)g) << 8 | ((int)b);
     *
     * unpack code:
     *
     * int rgb = ...;
     * uint8_t r = (rgb >> 16) & 0x0000ff;
     * uint8_t g = (rgb >> 8)  & 0x0000ff;
     * uint8_t b = (rgb)       & 0x0000ff;
     */
    union Color {
        /* Big-Endian
         * uint32_t r g b a
         *          | | | |
         * uint8_t  b g r a
         */
        struct Channel {
            uint8_t b;
            uint8_t g;
            uint8_t r;
            uint8_t a;
            Channel() : b(0U), g(0U), r(0U), a(0xffU) {}
            Channel(const uint8_t& b_, const uint8_t& g_, const uint8_t& r_) : b(b_), g(g_), r(r_), a(0xffU) {}
        };
        uint32_t rgba;
        Channel channel;
        Color() : channel() {}
        explicit Color(const uint32_t rgba_) : rgba(rgba_) {} explicit
        Color(const uint8_t& b_, const uint8_t& g_, const uint8_t& r_)
            : channel(b_, g_, r_) {}
    };
    Color color;
    PointXYZRGB() : PointXYZ(), color() {}
    PointXYZRGB(const float& x_, const float& y_, const float& z_,
    const uint8_t& b_, const uint8_t& g_, const uint8_t& r_)
        : PointXYZ(x_, y_, z_), color(b_, g_, r_)
    {}
};
template<typename PointT> struct PointCloud {
    Header header;
    uint32_t width;
    uint32_t height;
    bool isDense;
    ara::core::Vector<PointT> points;
    PointCloud<PointT>() : header(), width(0U), height(0U), isDense(false), points() {}
    PointCloud<PointT>(const Header& vHeader, const uint32_t& vWidth, const uint32_t& vHeight,
    const bool& vIsDense, const ara::core::Vector<PointT>& vPoints)
        : header(vHeader), width(vWidth), height(vHeight), isDense(vIsDense), points(vPoints) {}
};

template struct PointCloud<PointXYZ>;

template struct PointCloud<PointXYZI>;

template struct PointCloud<PointXYZIR>;

template struct PointCloud<PointXYZRGB>;

template<typename PointT> ara::core::Vector<PointField> &VizConvertToPointField()
{
    static ara::core::Vector<PointField> fields;
    return fields;
}
template<> ara::core::Vector<PointField> &VizConvertToPointField<PointXYZI>();
template<> ara::core::Vector<PointField> &VizConvertToPointField<PointXYZIR>();
template<> ara::core::Vector<PointField> &VizConvertToPointField<PointXYZ>();
template<> ara::core::Vector<PointField> &VizConvertToPointField<PointXYZRGB>();

bool Transfer(uint8_t * const destbuffer, const uint32_t destCount,
              const uint8_t * const srcBuffer, const uint32_t count);

template<typename PointT> RetCode ConvertToStdPointCloud(const PointCloud<PointT> &cloud, PointCloud2 &vizPointCloud2)
{
    if ((cloud.width == 0U) && (cloud.height == 0U)) {
        vizPointCloud2.width = static_cast<uint32_t>(cloud.points.size());
        vizPointCloud2.height = 1U;
    } else {
        if (cloud.points.size() != (static_cast<uint64_t>(cloud.width) * static_cast<uint64_t>(cloud.height))) {
            return RetCode::SIZE_ERROR;
        } else {
            vizPointCloud2.height = cloud.height;
            vizPointCloud2.width = cloud.width;
        }
    }

    const auto dataSize = static_cast<uint32_t>(sizeof(PointT) * cloud.points.size());
    vizPointCloud2.data.resize(static_cast<unsigned long>(dataSize));

    if (dataSize != 0U) {
        const PointCloud<PointT> cloudP = cloud;
        if (!Transfer(static_cast<uint8_t *>(&vizPointCloud2.data[0U]), dataSize,
            reinterpret_cast<const uint8_t *>(&cloudP.points[0U]), dataSize)) {
            return RetCode::MEMCPY_COPY_ERROR;
        }
    }

    vizPointCloud2.fields.clear();
    vizPointCloud2.fields = VizConvertToPointField<PointT>();

    vizPointCloud2.header = cloud.header;
    vizPointCloud2.pointStep = static_cast<uint32_t>(sizeof(PointT));
    vizPointCloud2.rowStep = static_cast<uint32_t>(sizeof(PointT) * cloud.width);
    vizPointCloud2.isDense = cloud.isDense;
    return RetCode::VIZ_OK;
}
}
}

#endif // VIZ_POINTCLOUD2_H
