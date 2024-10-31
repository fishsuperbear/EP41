{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/UssMsgDataType/AreaDistanceDataType",
  "Namespace": "::mdc::areadistances",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssComMsgDataType/Header"
    },
    {
      "Name": "msgRollingCounter",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "zonePrecision",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "areaDistanceList",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssMsgDataType/AreaDistanceVec"
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
  "IsPlane": "false"
}