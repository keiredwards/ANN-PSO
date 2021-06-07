using System;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;

namespace ANN_PSO
{
    class Program
    {
        /// <summary>
        /// Defines parameters for and creates a new Particle Swarm of Artificial Neural Networks to discover optimal weights and biases for the ANN with defined structure
        /// in fitting the specified activation function to specified data set.
        /// </summary>

        static void Main(string[] args)
        {
            int swarmSize = 100;    //Number of ANNs, each representing a particle in the Particle Swarm
            int[] annStructure = new int[] { 2, 6, 6, 6, 1 };   //Structure of the ANN with each value representing the number of nodes within each layer
            int maxGenerations = 10000;     //Maximum Number of Generations that the PSO will produce
            int activationFunctions = 3;    //Specified Activation function to be fit to the data set. 0: NULL, 1: Sigmoid , 2:Hyperbolic Tangent, 3: Cosine, 4: Math.Exp(-((value * value) / 2)), 5: non-linear cubic
                                            //Possible in future to expand PSO to optimize for activationFunctions
            int numberOfInformantGroups = 5;    //Particles in PSO can inform other particles. This value specifies the number of informant groups within the PSO that inform each other (social weight)
            float inertiaWeight = 0.9f; //Pull of particles to continue moving in current trajectory
            float cognitiveWeight = 1.2f;   //Pull of particles towards its previous best    
            float socialWeight = 1.2f;  //Pull of particle towards informant group's best
            float globalWeight = 1.6f;  //Pull of particle toward global best

            PSO.path = @"c:\Program Files\Data\2in_xor.txt";    //Path of Data set
            PSO pso = new PSO(swarmSize, annStructure, maxGenerations, activationFunctions, inertiaWeight, cognitiveWeight, socialWeight, globalWeight, numberOfInformantGroups); //Creates new PSO with specified parameters.


            
        }
    }
}
