{
  "ShortName": "/HuaweiMDC/X86PlatformServiceInterface/CameraMsgDataType/CameraEncodedMbufStruct",
  "Namespace": "::ara::camerax86",
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
      "Name": "CameraHeader",
      "Type": "/HuaweiMDC/PlatformServiceInterface/CameraMsgDataType/CameraHeaderMbufStruct"
    },
    {
      "Name": "RawData",
      "Type": "/HuaweiMDC/X86PlatformServiceInterface/CameraMsgDataType/Uint8VectorVector"
    }
  ],
  "IsPlane": "false"
}