using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assets.Scripts.VRGestures.Core
{
    /*
    public enum VRControllerGestureType
    {
        I,
        U,
        INFINITY,
        LessThan,
        None
    }*/

    public enum VRControllerGestureType
    {
        I,
        U,
        INFINITY,
        LessThan,
        O,
        L,
        e,
        M,
        Negative,
        None
    }
    public static class GestureTypeHelpers
    {
        public static string ToGestureString(this VRControllerGestureType gesture)
        {
            //string[] gestureNames = new string[] { "Inventory", "OK", "INFINITY", "Back", "None" };
            string[] gestureNames = new string[] { "Inventory", "OK", "INFINITY", "Back", "Lobby", "Lab", "Meeting Room", "Map", "Negative", "None" };
            return gestureNames[(int)gesture];
        }
    }
}
