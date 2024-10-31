{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/TrajetoryPredictionMsgDataType/ObjPredictionOutArray",
  "Namespace": "::ara::adsfi",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/FunctionalSoftware/EgoTrajectoryMsgDataType/Header"
    },
    {
      "Name": "isValid",
      "Type": "/HuaweiMDC/StdCppBasicDataType/bool"
    },
    {
      "Name": "predictionObstacle",
      "Type": "/HuaweiMDC/FunctionalSoftware/TrajetoryPredictionMsgDataType/PredictionObstacleVector"
    },
    {
      "Name": "startTime",
      "Type": "/HuaweiMDC/FunctionalSoftware/ObjectArrayMsgDataType/Time"
    },
    {
      "Name": "endTime",
      "Type": "/HuaweiMDC/FunctionalSoftware/ObjectArrayMsgDataType/Time"
    },
    {
      "Name": "changeOriginFlag",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint8_t"
    },
    {
      "Name": "selfIntent",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint8_t"
    },
    {
      "Name": "scenario",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint16_t"
    }
  ],
  "IsPlane": "false"
}