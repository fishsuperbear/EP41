{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/TrajetoryPredictionMsgDataType/PredictionObstacle",
  "Namespace": "::ara::adsfi",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "object",
      "Type": "/HuaweiMDC/FunctionalSoftware/TrajetoryPredictionMsgDataType/PerceptionObstacle"
    },
    {
      "Name": "gpsTime",
      "Type": "/HuaweiMDC/FunctionalSoftware/ObjectArrayMsgDataType/Time"
    },
    {
      "Name": "predictedPeriod",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "trajectory",
      "Type": "/HuaweiMDC/FunctionalSoftware/TrajetoryPredictionMsgDataType/TrajectoryInPredictionVector"
    },
    {
      "Name": "intent",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint8_t"
    },
    {
      "Name": "priority",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint8_t"
    },
    {
      "Name": "isStatic",
      "Type": "/HuaweiMDC/StdCppBasicDataType/bool"
    },
    {
      "Name": "feature",
      "Type": "/HuaweiMDC/FunctionalSoftware/TrajetoryPredictionMsgDataType/ObstacleFeature"
    }
  ],
  "IsPlane": "false"
}