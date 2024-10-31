{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/UssMsgDataType/DetectDistanceInfo",
  "Namespace": "::mdc::detectdistances",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "id",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "zoneDistance",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16",
      "Padding": "4"
    },
    {
      "Name": "zoneDistanceTimeStamp",
      "Type": "/HuaweiMDC/StdCppDataType/UInt64"
    },
    {
      "Name": "directDistance",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16",
      "Padding": "6"
    },
    {
      "Name": "directDistanceTimeStamp",
      "Type": "/HuaweiMDC/StdCppDataType/UInt64"
    },
    {
      "Name": "blind",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8",
      "Padding": "1"
    },
    {
      "Name": "crossDistanceLeft",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "crossDistanceRight",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16",
      "Padding": "2"
    },
    {
      "Name": "crossDistanceLeftTimeStamp",
      "Type": "/HuaweiMDC/StdCppDataType/UInt64"
    },
    {
      "Name": "crossDistanceRightTimeStamp",
      "Type": "/HuaweiMDC/StdCppDataType/UInt64"
    },
    {
      "Name": "confidence",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "sonarCounter",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8",
      "Padding": "2"
    },
    {
      "Name": "aftershockTime",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "reserveMsg1",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "reserveMsg2",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    }
  ],
  "IsPlane": "true"
}