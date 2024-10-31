{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/CameraMsgDataType/CameraPublishEncodedStruct",
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
      "Type": "/HuaweiMDC/PlatformServiceInterface/CameraMsgDataType/PointDataStruct"
    },
    {
      "Name": "CameraHeader",
      "Type": "/HuaweiMDC/PlatformServiceInterface/CameraMsgDataType/CameraHeaderStruct"
    }
  ],
  "IsPlane": "false"
}