{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/CameraMsgDataType/CameraEncodedMbufStruct",
  "Namespace": "::ara::camera",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "DataSize",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "SendTimeHigh",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "SendTimeLow",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "FrameType",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "VideoFormat",
      "Type": "/HuaweiMDC/StdCppDataType/String"
    },
    {
      "Name": "RawData",
      "Type": "/HuaweiMDC/StdCppDataType/rawBuffer"
    },
    {
      "Name": "CameraHeader",
      "Type": "/HuaweiMDC/PlatformServiceInterface/CameraMsgDataType/CameraHeaderMbufStruct"
    }
  ],
  "IsPlane": "false"
}