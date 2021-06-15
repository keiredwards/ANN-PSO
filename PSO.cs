using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text.RegularExpressions;


namespace ANN_PSO
{
    class Pso
    {
        private static readonly List<List<double>> VelocityChange = new List<List<double>>();
        private static readonly List<List<double>> Velocity = new List<List<double>>();

        public static List<List<double>> SwarmPositions = new List<List<double>>();
        private static readonly List<List<double>> InformantBestPositions = new List<List<double>>();
        private static readonly List<List<double>> ParticleBestPositions = new List<List<double>>();

        private static double _bestError = double.MaxValue; //Sets current best Error to max value
        private static readonly List<double> PersonalBestError = new List<double>();
        private static readonly List<double> CurrentError = new List<double>();
        private static readonly List<double> BestInformantError = new List<double>();

        public static List<double> Input = new List<double>();
        public static List<List<double>> Output = new List<List<double>>();
        private static readonly List<double> CorrectOutput = new List<double>();

        private static double[][] _biases;
        private static double[][][][] _weights;

        public static string Path;
        private static int _bestNet;

        public Pso(int swarmSize, int[] annStructure, int maxGenerations, int activationFunction, double inertiaWeight,
            double cognitiveWeight, double socialWeight, double globalWeight, int numberOfInformantGroups)
        {
            _biases = new double[swarmSize][];
            _weights = new double[swarmSize][][][];
            double[] particleBestError = new double[swarmSize];

            Random rnd = new Random();
            for (int particleNum = 0; particleNum < swarmSize; particleNum++)  
            {
                particleBestError[particleNum] = double.MaxValue;
                PersonalBestError.Add(double.MaxValue); //Sets best Errors to be max Value so any Error following is considered an improvement
                CurrentError.Add(double.MaxValue);
                SwarmPositions.Add(new List<double>());
                InformantBestPositions.Add(new List<double>());
                ParticleBestPositions.Add(new List<double>());
                Velocity.Add(new List<double>());
                VelocityChange.Add(new List<double>());
                Output.Add(new List<double>());
            }

            for (int i = 0; i < numberOfInformantGroups; i++)
            {
                BestInformantError.Add(double.MaxValue);
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
                        new Ann(particleNum, null, null, activationFunction, annStructure);
                        Velocity[particleNum].Clear();
                        Velocity[particleNum].AddRange(MultiplyList(rnd.NextDouble(), SwarmPositions[particleNum])); //give particle a random velocity created by multiplying its position by a random value.
                        ParticleBestPositions[particleNum].AddRange(SwarmPositions[particleNum]); //each position becomes the best ever position for that particle
                    }
                    else
                    {
                        SwarmPositions[particleNum].Clear();
                        new Ann(particleNum, _weights[particleNum], _biases[particleNum], activationFunction, annStructure);
                    }
                }
                //Assess and Update Particles based on results of previous generation
                for (int particleNumber = 0; particleNumber < swarmSize; particleNumber++)
                {
                    AssessFitness(Output[particleNumber], particleNumber, currentGeneration, numberOfInformantGroups);
                    const double temp = 0.5 / 1000;
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

            string inputText = System.IO.File.ReadAllText(Path);
            string reformattedInputText = Regex.Replace(inputText, @"\s+", " ");   //Fixes formatting of File to make all whitespace " "
            string[] inputSplit = reformattedInputText.Split(' '); 

            int numCount = 0;

            foreach (string value in inputSplit)  //Determines if value is an input or realOutput by the number of inputs for the ANN (annStructure[0]) size.
            {

                if (value != string.Empty)
                {
                   
                    if (numCount < annStructure[0])    
                    {
                        Input.Add(Convert.ToDouble(value));
                        numCount++;
                    }
                    else
                    {
                        CorrectOutput.Add(Convert.ToDouble(value));
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
            _biases[particleNumber] = new double[annStructure.Length - 1];
            var rnd = new Random();

            //Adds randomness to Weights for more varying exploration
            double b = rnd.NextDouble() * cognitiveWeight;
            double c = rnd.NextDouble() * socialWeight;
            double d = rnd.NextDouble() * globalWeight;

            //Calculates Velocity change based on PSO formula
            VelocityChange[particleNumber].Clear();
            for (int x = 0; x < SwarmPositions[particleNumber].Count - 1; x++)
            {
                VelocityChange[particleNumber].Add((inertiaWeight * (Velocity[particleNumber][x])) + (b * (ParticleBestPositions[particleNumber][x] - SwarmPositions[particleNumber][x])) + (c * (InformantBestPositions[informantGroup][x] - SwarmPositions[particleNumber][x])) + (d * (ParticleBestPositions[_bestNet][x] - SwarmPositions[particleNumber][x])));
                Velocity[particleNumber][x] = VelocityChange[particleNumber][x];
            }

            //Adjusts position of particles based on their new velocities
            for (int x = 0; x < SwarmPositions[particleNumber].Count - 1; x++)
            {
                SwarmPositions[particleNumber][x] = (SwarmPositions[particleNumber][x] + (stepSize * Velocity[particleNumber][x]));
            }

            int particlePosLength = SwarmPositions[particleNumber].Count;
            int weightPos = 0;
            
            //Creates Weights for ANN
            _weights[particleNumber] = new double[annStructure.Length - 1][][];
            for (int layerNum = 0; layerNum < annStructure.Length - 1; layerNum++)
            {
                _weights[particleNumber][layerNum] = new double[annStructure[layerNum]][];

                for (int neuron = 0; neuron < annStructure[layerNum]; neuron++)
                {
                    _weights[particleNumber][layerNum][neuron] = new double[annStructure[layerNum + 1]];

                }

            }

            for (int layerNum = 0; layerNum < annStructure.Length - 1; layerNum++)
            {
                //Adjusts Positions of particles so that if they exceed the investigation area they are repositioned to be at the edge of that area.
                SwarmPositions[particleNumber][particlePosLength - annStructure.Length - 1 + layerNum] = (SwarmPositions[particleNumber][particlePosLength - annStructure.Length - 1 + layerNum] > 10D) ? 10D : SwarmPositions[particleNumber][particlePosLength - annStructure.Length - 1 + layerNum];     
                SwarmPositions[particleNumber][particlePosLength - annStructure.Length - 1 + layerNum] = (SwarmPositions[particleNumber][particlePosLength - annStructure.Length - 1 + layerNum] < -10D) ? -10D : SwarmPositions[particleNumber][particlePosLength - annStructure.Length - 1 + layerNum];
                _biases[particleNumber][layerNum] = SwarmPositions[particleNumber][particlePosLength - annStructure.Length - 1 + layerNum];
                for (int neuron = 0; neuron < annStructure[layerNum]; neuron++)
                {
                    for (int neuronConnection = 0; neuronConnection < annStructure[layerNum + 1]; neuronConnection++)
                    {
                        _weights[particleNumber][layerNum][neuron][neuronConnection] = SwarmPositions[particleNumber][weightPos];
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
        private static void AssessFitness(IReadOnlyList<double> realOutput, int particleNumber, int generation, int numberOfInformantGroups)
        {
            double error = 0;
            int informantGroup = particleNumber % numberOfInformantGroups;
            CurrentError[particleNumber] = error;
            for (int x = 0; x < realOutput.Count && x < CorrectOutput.Count; x++)
            {
                error = Math.Pow((CorrectOutput[x] - realOutput[x]), 2);
                CurrentError[particleNumber] += error;
            }

            if (CurrentError[particleNumber] < PersonalBestError[particleNumber])   //New personal best
            {
                PersonalBestError[particleNumber] = CurrentError[particleNumber];
                ParticleBestPositions[particleNumber].Clear();
                ParticleBestPositions[particleNumber].AddRange(SwarmPositions[particleNumber]);

                if (CurrentError[particleNumber] < BestInformantError[informantGroup]) //New group best
                {
                    BestInformantError[informantGroup] = CurrentError[particleNumber];
                    InformantBestPositions[informantGroup].Clear();
                    InformantBestPositions[informantGroup].AddRange(SwarmPositions[particleNumber]);

                    if (CurrentError[particleNumber] < _bestError)   //New population best
                    {
                        _bestError = CurrentError[particleNumber];
                        _bestNet = particleNumber;
                        Debug.WriteLine("best error: " + _bestError + "  generation: " + generation); //Writes new Best Error to Console
                    }
                }
            }
        }
    }
}
