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

    //public enum VRControllerGestureType
    //{
    //    I,
    //    U,
    //    INFINITY,
    //    LessThan,
    //    O,
    //    L,
    //    e,
    //    M,
    //    Negative,
    //    None
    //}
    //public enum VRControllerGestureType
    //{
    //    L,
    //    e,
    //    S,
    //    GreaterThan,
    //    INFINITY,
    //    Negative,
    //    None
    //}

    //public enum VRControllerGestureType
    //{
    //    SwipeRight,
    //    SwipeLeft,
    //    GreaterThan,
    //    LessThan,
    //    Negative,
    //    None
    //}

    /*public enum VRControllerGestureType
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
    }*/

    public enum VRControllerGestureType
    {
        L,
        O,
        I,
        U,
        e,
        M,
        LessThan,
        INFINITY,
        Negative,
        None
    }

    /*    public enum VRControllerGestureType
         {
             L,
             O,
             V,
             e,
             S,
             D,
             C,
             M,
             GreaterThan,
             INFINITY,
             Negative,
             None
         }*/

    public static class GestureTypeHelpers
    {
        public static string ToGestureString(this VRControllerGestureType gesture)
        {
            //string[] gestureNames = new string[] { "Inventory", "OK", "INFINITY", "Back", "None" };
            //string[] gestureNames = new string[] { "Inventory", "OK", "INFINITY", "Back", "Lobby", "Lab", "Meeting Room", "Map", "Negative", "None" };
            //string[] gestureNames = new string[] { "L","e","S",">","INFINITY","Negative","None" };
            //string[] gestureNames = new string[] { "SwipeRight", "SwipeLeft", ">", "<", "Negative", "None" };
            string[] gestureNames = new string[] { "L", "O", "I", "U", "e", "M", "LessThan", "INFINITY", "Negative", "None" };
            //string[] gestureNames = new string[] { "I", "U", "INFINITY", "LessThan", "O", "L", "e", "M", "Negative", "None" };
            //string[] gestureNames = new string[] { "INFINITY", "LessThan", "M", "e", "U", "I", "O", "L", "Negative", "None" };
            //string[] gestureNames = new string[] { "L","O","V","e","S","D","C","M",">","INFINITY","Negative", "None" };
            return gestureNames[(int)gesture];
        }
    }
}
