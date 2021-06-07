using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Numerics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;


namespace ANN_PSO
{
    class PSO
    {



        private static List<List<double>> velocityChange = new List<List<double>>();
        private static List<List<double>> velocity = new List<List<double>>();


        public static List<List<double>> swarmPositions = new List<List<double>>();
        private static List<List<double>> newSwarmPositions = new List<List<double>>();
        private static List<List<double>> informantBestPositions = new List<List<double>>();
        private static List<List<double>> particleBestPositions = new List<List<double>>();

        private static double _bestError = double.MaxValue; //Sets current best Error to max value
        private static List<double> _personalBestError = new List<double>();
        private static List<double> _currentError = new List<double>();
        private static List<double> _bestInformantError = new List<double>();

        public static List<double> input = new List<double>();
        public static List<double> correctOutput = new List<double>();
        public static List<List<double>> output = new List<List<double>>();

        static double[][] biases;
        public static double[][][][] weights;


        public static string path;

        static int bestnet;

        public PSO(int swarmSize, int[] annStructure, int maxGenerations, int activationFunction, double inertiaWeight,
            double cognitiveWeight, double socialWeight, double globalWeight, int numberOfInformantGroups)
        {

            biases = new double[swarmSize][];
            weights = new double[swarmSize][][][];

            double[] particleBestError = new double[swarmSize];

            Random rnd = new Random();
            for (int particleNum = 0; particleNum < swarmSize; particleNum++)  
            {
                particleBestError[particleNum] = double.MaxValue;
                _personalBestError.Add(double.MaxValue); //Sets best Errors to be max Value so any Error following is considered an improvement
                _currentError.Add(double.MaxValue);
                swarmPositions.Add(new List<double>());
                newSwarmPositions.Add(new List<double>());
                informantBestPositions.Add(new List<double>());
                particleBestPositions.Add(new List<double>());
                velocity.Add(new List<double>());
                velocityChange.Add(new List<double>());
                output.Add(new List<double>());
            }


            for (int i = 0; i < numberOfInformantGroups; i++)
            {
                _bestInformantError.Add(double.MaxValue);
            }

            DataInputSort(annStructure);

            
            //Main Control Function of the PSO. Running through each generation
            int currentGeneration = 0;
            while (currentGeneration < maxGenerations)
            {
                for (int particleNum = 0; particleNum < swarmSize; particleNum++)
                {

                    if (currentGeneration == 0) //If this is the first generation of the PSO
                    {
                        new ANN(particleNum, null, null, activationFunction, annStructure);
                        velocity[particleNum].Clear();
                        velocity[particleNum].AddRange(MultiplyList(rnd.NextDouble(), swarmPositions[particleNum])); //give particle a random velocity created by multiplying its position by a random value.
                        particleBestPositions[particleNum].AddRange(swarmPositions[particleNum]); //each position becomes the best ever position for that particle

                    }
                    else
                    {
                        swarmPositions[particleNum].Clear();
                        new ANN(particleNum, weights[particleNum], biases[particleNum], activationFunction, annStructure);  //creates new ANN with updated values

                    }
                }

                //Assess and Update Particles based on results of previous generation
                for (int particleNumber = 0; particleNumber < swarmSize; particleNumber++)
                {
                    AssessFitness(output[particleNumber], particleNumber, currentGeneration, numberOfInformantGroups);
                    double temp = 0.5 / 1000;
                    if (inertiaWeight > 0.4) inertiaWeight -= temp; //Slowly decreases the inertia weight to allow particles to explore a more and more precise area in the solution space
                    UpdateParticle(inertiaWeight, cognitiveWeight, socialWeight, globalWeight, annStructure, particleNumber, numberOfInformantGroups);
                }

                currentGeneration++;
            }

        }

        /// <summary>
        /// Processes the Input Data File into suitable inputs for the ANN and the ANNs ideal realOutput.
        /// </summary>
        /// <param name="annStructure"> The Structure of the ANN. Represented by an array with the number of nodes in each layer</param>
        private static void DataInputSort(int[] annStructure)
        {
            if (annStructure == null) throw new ArgumentNullException(nameof(annStructure));

            string inputText = System.IO.File.ReadAllText(path);
            string reformattedInputText = Regex.Replace(inputText, @"\s+", " ");   //Fixes formatting of File to make all whitespace " "
            string[] inputSplit = reformattedInputText.Split(' '); 

            int numCount = 0;

            foreach (string value in inputSplit)  //Determines if value is an input or realOutput by the number of inputs for the ANN (annStructure[0]) size.
            {

                if (value != string.Empty)
                {
                   
                    if (numCount < annStructure[0])    
                    {
                        input.Add(Convert.ToDouble(value));
                        numCount++;
                    }
                    else
                    {
                        correctOutput.Add(Convert.ToDouble(value));
                        numCount = 0;
                    }
                }
            }
        }

        /// <summary>
        /// Updates Particle to find its new position based on its current velocity,current error, its best error, its informants best errors and the global best error.
        /// </summary>
        /// <param name="inertiaWeight">How much of the velocity of the particle is carried onto its next iteration</param>
        /// <param name="cognitiveWeight">The pull of the particle towards its best ever position</param>
        /// <param name="socialWeight">The pull of the particle towards its informant's best ever position</param>
        /// <param name="globalWeight">The pull of the particle towards the global best ever position</param>
        /// <param name="annStructure">The Layout of the ANN</param>
        /// <param name="particleNumber">The ANN's index.</param>
        /// <param name="numberOfInformantGroups">Number of Groups that the Population is split into for a particle's social pull</param>
        private static void UpdateParticle(double inertiaWeight, double cognitiveWeight, double socialWeight, double globalWeight, int[] annStructure, int particleNumber, int numberOfInformantGroups)
        {
            double stepSize = 1;
            var informantGroup = particleNumber % numberOfInformantGroups;

            biases[particleNumber] = new double[annStructure.Length - 1];
            var rnd = new Random();

            //Adds randomness to Weights for more varying exploration
            double b = rnd.NextDouble() * cognitiveWeight;
            double c = rnd.NextDouble() * socialWeight;
            double d = rnd.NextDouble() * globalWeight;



            //Calculates Velocity change based on PSO formula
            velocityChange[particleNumber].Clear();
            for (int x = 0; x < swarmPositions[particleNumber].Count - 1; x++)
            {
                velocityChange[particleNumber].Add((inertiaWeight * (velocity[particleNumber][x])) + (b * (particleBestPositions[particleNumber][x] - swarmPositions[particleNumber][x])) + (c * (informantBestPositions[informantGroup][x] - swarmPositions[particleNumber][x])) + (d * (particleBestPositions[bestnet][x] - swarmPositions[particleNumber][x])));
                velocity[particleNumber][x] = velocityChange[particleNumber][x];
            }

            //Adjusts position of particles based on their new velocities
            for (int x = 0; x < swarmPositions[particleNumber].Count - 1; x++)
            {
                swarmPositions[particleNumber][x] = (swarmPositions[particleNumber][x] + (stepSize * velocity[particleNumber][x]));
            }

            int particlePosLength = swarmPositions[particleNumber].Count;
            int weightPos = 0;
            
            //Creates Weights for ANN
            weights[particleNumber] = new double[annStructure.Length - 1][][];
            for (int layerNum = 0; layerNum < annStructure.Length - 1; layerNum++)
            {
                weights[particleNumber][layerNum] = new double[annStructure[layerNum]][];

                for (int neuron = 0; neuron < annStructure[layerNum]; neuron++)
                {
                    weights[particleNumber][layerNum][neuron] = new double[annStructure[layerNum + 1]];

                }

            }

            for (int layerNum = 0; layerNum < annStructure.Length - 1; layerNum++)
            {
                //Adjusts Positions of particles so that if they exceed the investigation area they are repositioned to be at the edge of that area.
                swarmPositions[particleNumber][particlePosLength - annStructure.Length - 1 + layerNum] = (swarmPositions[particleNumber][particlePosLength - annStructure.Length - 1 + layerNum] > 10D) ? 10D : swarmPositions[particleNumber][particlePosLength - annStructure.Length - 1 + layerNum];     
                swarmPositions[particleNumber][particlePosLength - annStructure.Length - 1 + layerNum] = (swarmPositions[particleNumber][particlePosLength - annStructure.Length - 1 + layerNum] < -10D) ? -10D : swarmPositions[particleNumber][particlePosLength - annStructure.Length - 1 + layerNum];
                biases[particleNumber][layerNum] = swarmPositions[particleNumber][particlePosLength - annStructure.Length - 1 + layerNum];

                for (int neuron = 0; neuron < annStructure[layerNum]; neuron++)
                {
                    for (int neuronConnection = 0; neuronConnection < annStructure[layerNum + 1]; neuronConnection++)
                    {
                        weights[particleNumber][layerNum][neuron][neuronConnection] = swarmPositions[particleNumber][weightPos];
                        weightPos++;
                    }
                }
            }
        }

        /// <summary>
        /// Multiplies Each value in the list by a singular double value.
        /// </summary>
        /// <param name="multiplier">Value Multiplier</param>
        /// <param name="oldList">List to be multiplied</param>
        /// <returns></returns>
        private static List<double> MultiplyList(double multiplier, List<double> oldList)
        {
            List<double> newList = new List<double>();
            foreach (var listEntry in oldList)
            {
                newList.Add((listEntry * multiplier));
            }

            return newList;
        }


        /// <summary>
        /// Assesses the Fitness of the Particle based on the realOutput given and the correct realOutput as found in the text file.
        /// </summary>
        /// <param name="realOutput">Output from the ANN</param>
        /// <param name="particleNumber">The number of the particle/ANN being assessed.</param>
        /// <param name="generation">Which maxGenerations of ANN this is. </param>
        /// <param name="numberOfInformantGroups">Number of Groups that the Population is split into for a particle's social pull</param>
        /// <returns></returns>
        /// 
        static double AssessFitness(List<double> realOutput, int particleNumber, int generation, int numberOfInformantGroups)
        {
            double Error = 0;
            int informantGroup = particleNumber % numberOfInformantGroups;
            _currentError[particleNumber] = Error;

            for (int x = 0; x < realOutput.Count && x < correctOutput.Count; x++)
            {
                Error = Math.Pow((correctOutput[x] - realOutput[x]), 2);

                _currentError[particleNumber] += Error;

            }

            if (_currentError[particleNumber] < _personalBestError[particleNumber])   //New personal best
            {
                _personalBestError[particleNumber] = _currentError[particleNumber];
                particleBestPositions[particleNumber].Clear();
                particleBestPositions[particleNumber].AddRange(swarmPositions[particleNumber]);

                if (_currentError[particleNumber] < _bestInformantError[informantGroup]) //New group best
                {
                    _bestInformantError[informantGroup] = _currentError[particleNumber];
                    informantBestPositions[informantGroup].Clear();
                    informantBestPositions[informantGroup].AddRange(swarmPositions[particleNumber]);

                    if (_currentError[particleNumber] < _bestError)   //New population best
                    {
                        _bestError = _currentError[particleNumber];
                        bestnet = particleNumber;

                        Debug.WriteLine("best error: " + _bestError + "  generation: " + generation); //Writes new Best Error to Console
                    }
                }
            }

           

            return _bestError / realOutput.Count;

        }
    }
}
