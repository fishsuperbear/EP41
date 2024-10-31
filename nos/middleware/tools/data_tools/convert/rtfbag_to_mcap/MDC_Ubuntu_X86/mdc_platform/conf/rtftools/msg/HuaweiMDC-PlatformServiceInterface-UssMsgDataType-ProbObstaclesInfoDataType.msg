{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/UssMsgDataType/ProbObstaclesInfoDataType",
  "Namespace": "::mdc::probobstacles",
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
      "Name": "msgRollingCounter1",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "msgRollingCounter2",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "msgRollingCounter3",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "msgRollingCounter4",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "probObstacleList",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssMsgDataType/ProbObstaclesInfoVec"
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
      "Name": "pasObjOutputSts",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "timeStamp",
      "Type": "/HuaweiMDC/StdCppDataType/UInt64"
    },
    {
      "Name": "psolTimeStamp",
      "Type": "/HuaweiMDC/StdCppDataType/UInt64"
    }
  ],
  "IsPlane": "false"
}