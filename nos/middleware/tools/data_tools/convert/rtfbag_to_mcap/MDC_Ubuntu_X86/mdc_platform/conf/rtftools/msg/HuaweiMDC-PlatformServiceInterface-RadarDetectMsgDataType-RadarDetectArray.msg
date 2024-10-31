{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/RadarDetectMsgDataType/RadarDetectArray",
  "Namespace": "::ara::rdetect",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/RadarMsgDataType/Header"
    },
    {
      "Name": "sensorId",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "radarState",
      "Type": "/HuaweiMDC/PlatformServiceInterface/RadarMsgDataType/RadarState"
    },
    {
      "Name": "detectList",
      "Type": "/HuaweiMDC/PlatformServiceInterface/RadarDetectMsgDataType/DetectList"
    },
    {
      "Name": "maxDistanceOverAzimuthList",
      "Type": "/HuaweiMDC/PlatformServiceInterface/RadarMsgDataType/FloatVector"
    },
    {
      "Name": "azimuthForMaxDistanceList",
      "Type": "/HuaweiMDC/PlatformServiceInterface/RadarMsgDataType/FloatVector"
    },
    {
      "Name": "factorDistanceOverElevationList",
      "Type": "/HuaweiMDC/PlatformServiceInterface/RadarMsgDataType/FloatVector"
    },
    {
      "Name": "elevationForFactorMaxDistanceList",
      "Type": "/HuaweiMDC/PlatformServiceInterface/RadarMsgDataType/FloatVector"
    },
    {
      "Name": "maxDistanceDueToProgram",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "minDistanceDueToProgram",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    }
  ],
  "IsPlane": "false"
}