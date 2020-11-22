using System;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;

namespace ANN_PSO
{
    class Program
    {
        static void Main(string[] args)
        {
            int swarmSize = 100;
            int[] annStructure = new int[] { 2, 6, 6, 6, 1 };
            int epochs = 10000;
            int activationFunctions = 3;
            int numberOfInformantGroups = 5;

            PSO.path = @"c:\Program Files\Data\2in_xor.txt";
            PSO pso = new PSO(swarmSize, annStructure, epochs, activationFunctions, 0.9, 1.2, 1.2, 1.6, numberOfInformantGroups);


            
        }
    }
}
