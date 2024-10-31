{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/ImuInfoDataType/ImuInfo",
  "Namespace": "::ara::imu",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/Header"
    },
    {
      "Name": "angularVelocity",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/GeometryPoit"
    },
    {
      "Name": "acceleration",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/GeometryPoit"
    },
    {
      "Name": "imuStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "temperature",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    }
  ],
  "IsPlane": "false"
}