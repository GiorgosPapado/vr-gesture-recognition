using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using UnityEngine;
using Assets.Scripts.VRGestures.Core.Input;

namespace Assets.Scripts.VRGestures.Core.ONNX
{
    public class MLNETVRControllerGestureInference : MonoBehaviour, IVRGestureInferenceEngine
    {
        private InferenceSession m_Session;
        public Handedness m_Handedness = Handedness.RIGHT;
        
        [SerializeField] private string m_ModelFileName = "vr-controller-gestrec-ext-wneg.onnx";
        // Optional absolute override; leave empty to load from StreamingAssets/gestures/
        [SerializeField] private string m_ModelPathOverride = "";

        public void Awake()
        {
            string path_to_onnx_model = string.IsNullOrEmpty(m_ModelPathOverride)
                ? System.IO.Path.Combine(Application.streamingAssetsPath, "gestures", m_ModelFileName)
                : m_ModelPathOverride;

            m_Session = new InferenceSession(path_to_onnx_model);
        }
        public (VRControllerGestureType Gesture, float Confidence) Inference(VRGestureSample gesture)
        {
            int ndim = 4;
            var gr = m_Handedness == Handedness.RIGHT ? gesture.RightControllerData : gesture.LeftControllerData;           
            Tensor<float> inputTensor = new DenseTensor<float>(new int[] { 1, ndim, gr.Count });
            Vector3 ref_pos = gr[0].Position;
            Quaternion inv_ref_orient = Quaternion.Inverse(gr[0].Orientation);
            for (int i = 0; i < gr.Count; i++)
            {
                int feat_index = 0;
                Vector3 trans_pos = inv_ref_orient * (gr[i].Position - ref_pos);
                Quaternion trans_orient = inv_ref_orient * gr[i].Orientation;
                //Vector3 trans_pos = gr[i].Position;
                inputTensor[0, feat_index++, i] = trans_pos.x;
                inputTensor[0, feat_index++, i] = trans_pos.y;
                inputTensor[0, feat_index++, i] = trans_pos.z;
                if(ndim == 4) { 
                    inputTensor[0, feat_index++, i] = gr[i].Time;
                }
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            using (var results = m_Session.Run(inputs))
            {
                var q = results.First().AsTensor<float>().ToArray();
                var (conf, label) = results.First().AsTensor<float>().ToArray().Select((val, index) => (val, index)).Max();                
                return ((VRControllerGestureType)label, conf);
            }
        }
    }
}
