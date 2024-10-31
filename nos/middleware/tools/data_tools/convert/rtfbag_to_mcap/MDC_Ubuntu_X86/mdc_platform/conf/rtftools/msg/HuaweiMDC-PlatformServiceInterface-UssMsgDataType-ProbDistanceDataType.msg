{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/UssMsgDataType/ProbDistanceDataType",
  "Namespace": "::mdc::probdistances",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssComMsgDataType/Header"
    },
    {
      "Name": "version",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssComMsgDataType/Version"
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
      "Name": "probDistanceInfoList",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssMsgDataType/ProbDistanceVec"
    },
    {
      "Name": "echoInfoList",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssMsgDataType/ProbEchoDistanceVec"
    },
    {
      "Name": "odoInfo",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssMsgDataType/ProbOdoInfo"
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
    },
    {
      "Name": "timeStamp",
      "Type": "/HuaweiMDC/StdCppDataType/UInt64"
    }
  ],
  "IsPlane": "false"
}