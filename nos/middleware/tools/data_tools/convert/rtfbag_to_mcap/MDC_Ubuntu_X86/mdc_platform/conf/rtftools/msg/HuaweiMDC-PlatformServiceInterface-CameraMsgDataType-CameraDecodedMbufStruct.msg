{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/CameraMsgDataType/CameraDecodedMbufStruct",
  "Namespace": "::ara::camera",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "Height",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "Width",
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
      "Name": "DataSize",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
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