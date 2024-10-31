{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/UssCommonMsgDataType/DistancesDataType",
  "Namespace": "::mdc::commondistances",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssComMsgDataType/Header"
    },
    {
      "Name": "msgRollingCounterFront",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "msgRollingCounterRear",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "distanceInfoList",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssCommonMsgDataType/DistancesVec"
    },
    {
      "Name": "frontNearBrrDist",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "rearNearBrrDist",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "frontSoundIndication",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "rearSoundIndication",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "sideSoundIndication",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
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