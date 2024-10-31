{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/GnssInfo",
  "Namespace": "::ara::gnss",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/Header"
    },
    {
      "Name": "longitude",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "latitude",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "elevation",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "utmPosition",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/GeometryPoit"
    },
    {
      "Name": "utmZoneNum",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    },
    {
      "Name": "utmZoneChar",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "attitude",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/GeometryPoit"
    },
    {
      "Name": "linearVelocity",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/GeometryPoit"
    },
    {
      "Name": "sdPosition",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/GeometryPoit"
    },
    {
      "Name": "sdVelocity",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/GeometryPoit"
    },
    {
      "Name": "sdAttitude",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/GeometryPoit"
    },
    {
      "Name": "cep68",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "cep95",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "second",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "satUseNum",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    },
    {
      "Name": "satInViewNum",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    },
    {
      "Name": "solutionStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "positionType",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "pDop",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "hDop",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "vDop",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "attitudeDual",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/GeometryPoit"
    },
    {
      "Name": "sdAngleDual",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/GeometryPoit"
    },
    {
      "Name": "baseLineLengthDual",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "solutionStatusDual",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    },
    {
      "Name": "positionTypeDual",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    },
    {
      "Name": "solutionSourceDual",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    }
  ],
  "IsPlane": "false"
}