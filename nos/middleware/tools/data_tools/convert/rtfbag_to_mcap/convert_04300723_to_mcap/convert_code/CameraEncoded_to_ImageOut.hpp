#pragma once
#include "ara/camera/impl_type_cameraencodedmbufstruct.h"
#include "proto/soc/sensor_image.pb.h"  // proto 数据变量

struct CameraEncodedMbufStructHelp {
    ::UInt32 MbufCount;                               //mbuf个数
    ::UInt32 PrivateSize;                             //mbuf0private数据长度
    ara::camera::CameraEncodedMbufStruct RealStruct;  //mbuf0private数据
    std::vector<uint8_t> MbufData;                    //mbuf0数据区数据

    static bool IsPlane() { return false; }

    using IsDpRawDataTag = void;
    using IsEnumerableTag = void;

    template <typename F>
    void enumerate(F& fun) {
        fun(MbufCount);
        fun(PrivateSize);
        fun(RealStruct);
        fun(MbufData);
    }

    template <typename F>
    void enumerate(F& fun) const {
        fun(MbufCount);
        fun(PrivateSize);
        fun(RealStruct);
        fun(MbufData);
    }

    bool operator==(const CameraEncodedMbufStructHelp& t) const { return (MbufCount == t.MbufCount) && (PrivateSize == t.PrivateSize) && (RealStruct == t.RealStruct) && (MbufData == t.MbufData); }
};

hozon::soc::CompressedImage CameraEncodedFrameToImageOut(CameraEncodedMbufStructHelp mdc_data) {
    hozon::soc::CompressedImage proto_data;
    uint32_t ftype = 0;

    // hozon.perception.datacollection.Header
    proto_data.mutable_header()->set_seq(mdc_data.RealStruct.CameraHeader.Seq);
    proto_data.mutable_header()->set_frame_id(mdc_data.RealStruct.CameraHeader.FrameId);
    proto_data.set_measurement_time(static_cast<double>(mdc_data.RealStruct.CameraHeader.Stamp.Sec) + static_cast<double>(mdc_data.RealStruct.CameraHeader.Stamp.Nsec) / 1e9);
    if (mdc_data.RealStruct.FrameType == 19) {
        // I frame
        ftype = 1;
    } else if (mdc_data.RealStruct.FrameType == 1) {
        // P frame
        ftype = 2;
    }
    proto_data.set_frame_type(ftype);

    // data
    proto_data.set_data(std::string(mdc_data.MbufData.begin(), mdc_data.MbufData.end()));

    // length
    proto_data.set_length(mdc_data.MbufData.size());

    return proto_data;
}