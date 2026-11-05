using Assets.Scripts.VRGestures.Core.ONNX;
using Assets.Scripts.VRGestures.Core.Input;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Assets.Scripts.VRGestures.Core
{
    public class VRControllerGestureInput : MonoBehaviour
    {
        private static IVRGestureInferenceEngine m_Inference;
        private static VRControllerGestureType m_LastGesture = VRControllerGestureType.None;
        private static float m_LastGestureConfidence = 0.0f;
        public float m_ConfidenceThreshold = 0.9f;

        public void Start()
        {
            m_Inference = GetComponent<IVRGestureInferenceEngine>();
        }

        public void OnVRControllerGestureComplete(List<SensorDataBatch> sensorData)
        {
            VRGestureSample sample = new VRGestureSample();
            sample.HeadsetData.AddRange(sensorData.Select(x => x.Headset));
            sample.LeftControllerData.AddRange(sensorData.Select(x => x.LeftController));
            sample.RightControllerData.AddRange(sensorData.Select(x => x.RightController));
            var (gesture, conf) = m_Inference.Inference(sample);
            //if(conf > m_ConfidenceThreshold)
            if(true)
            {
                m_LastGesture = gesture;
                m_LastGestureConfidence = conf;
            }
            else
            {
                m_LastGesture = VRControllerGestureType.Negative;
                //m_LastGestureConfidence = 1.0f - conf;
                m_LastGestureConfidence = conf;
            }
        }

        public static bool GetGesture(VRControllerGestureType gestureType)
        {
            return gestureType == m_LastGesture;
        }

        public static (VRControllerGestureType Gesture, float Confidence) GetLastGesture()
        {
            return (m_LastGesture, m_LastGestureConfidence);
        }

        public void LateUpdate()
        {
            m_LastGesture = VRControllerGestureType.None;
            m_LastGestureConfidence = 1.0f;
        }
    }
}
