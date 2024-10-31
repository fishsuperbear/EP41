{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/TrajetoryPredictionMsgDataType/ObstacleFeature",
  "Namespace": "::ara::adsfi",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "id",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint32_t"
    },
    {
      "Name": "polygonPoint",
      "Type": "/HuaweiMDC/ApplicationDataType/PointArray"
    },
    {
      "Name": "position",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "frontPosition",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "velocity",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "rawVelocity",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "acceleration",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "velocityHeading",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "speed",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "acc",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "theta",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "length",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "width",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "height",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "trackingTime",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "timestamp",
      "Type": "/HuaweiMDC/FunctionalSoftware/ObjectArrayMsgDataType/Time"
    },
    {
      "Name": "tPosition",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "tVelocity",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "tVelocityHeading",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "tSpeed",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "tAcceleration",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "tAcc",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "isStill",
      "Type": "/HuaweiMDC/StdCppBasicDataType/bool"
    },
    {
      "Name": "type",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint8_t"
    },
    {
      "Name": "labelUpdateTimeDelta",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "priority",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint8_t"
    },
    {
      "Name": "isNearJunction",
      "Type": "/HuaweiMDC/StdCppBasicDataType/bool"
    },
    {
      "Name": "futureTrajectoryPoints",
      "Type": "/HuaweiMDC/FunctionalSoftware/TrajetoryPredictionMsgDataType/PredictionTrajectoryPointVector"
    },
    {
      "Name": "shortTermPredictedTrajectoryPoints",
      "Type": "/HuaweiMDC/FunctionalSoftware/TrajetoryPredictionMsgDataType/TrajectoryPointVector"
    },
    {
      "Name": "predictedTrajectory",
      "Type": "/HuaweiMDC/FunctionalSoftware/TrajetoryPredictionMsgDataType/TrajectoryInPredictionVector"
    },
    {
      "Name": "adcTrajectoryPoint",
      "Type": "/HuaweiMDC/FunctionalSoftware/TrajetoryPredictionMsgDataType/TrajectoryPointVector"
    }
  ],
  "IsPlane": "false"
}