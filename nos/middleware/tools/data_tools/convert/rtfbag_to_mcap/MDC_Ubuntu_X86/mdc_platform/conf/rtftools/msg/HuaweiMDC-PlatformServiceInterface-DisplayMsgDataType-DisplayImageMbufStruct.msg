{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/DisplayMsgDataType/DisplayImageMbufStruct",
  "Namespace": "::ara::display",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "Height",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint32_t"
    },
    {
      "Name": "Width",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint32_t"
    },
    {
      "Name": "SendTimeStamp",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint64_t"
    },
    {
      "Name": "DataSize",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint32_t",
      "Padding": "4"
    },
    {
      "Name": "RawData",
      "Type": "/HuaweiMDC/StdCppDataType/rawBuffer"
    },
    {
      "Name": "Seq",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint32_t"
    },
    {
      "Name": "ChannelId",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint8_t"
    },
    {
      "Name": "InputFormat",
      "Type": "/HuaweiMDC/PlatformServiceInterface/DisplayMsgDataType/ImageFormatTypeE"
    },
    {
      "Name": "OutputFormat",
      "Type": "/HuaweiMDC/PlatformServiceInterface/DisplayMsgDataType/ImageFormatTypeE"
    },
    {
      "Name": "RotateMode",
      "Type": "/HuaweiMDC/PlatformServiceInterface/DisplayMsgDataType/RotateModeE"
    }
  ],
  "IsPlane": "true"
}