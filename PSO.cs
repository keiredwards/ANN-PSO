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
        static double[][][] particleBestLocation;
        public static double bestError = double.MaxValue; //Sets current best Error to max value
        static double[][] bestLocations;
        public static List<double> position = new List<double>();
        public static List<List<double>> velocityChange = new List<List<double>>();
        public static List<List<double>> velocity = new List<List<double>>();
        public static List<List<double>> swarmPositions = new List<List<double>>();
        public static List<List<double>> newSwarmPositions = new List<List<double>>();
        public static List<List<double>> informantBestPositions = new List<List<double>>();
        public static List<List<double>> particleBestPositions = new List<List<double>>();
        public static List<double> personalBestError = new List<double>();
        public static List<double> currentError = new List<double>();
        public static List<double> bestInformantErrror = new List<double>();
        public static List<double> input = new List<double>();

        static double[][] biases;
        public static double[][][][] weights;
        public static List<double> correctOutput = new List<double>();
        public static List<List<double>> output = new List<List<double>>();
        public static List<int> informants = new List<int>();

        public static string path;

        static int bestnet;

        public PSO(int swarmSize, int[] annStructure, int maxGenerations, int activationFunction, double inertiaWeight,
            double cognitiveWeight, double socialWeight, double globalWeight, int numberOfInformantGroups)
        {

            bestLocations = new double[annStructure.Length][];

            particleBestLocation = new double[swarmSize][][];
            biases = new double[swarmSize][];
            weights = new double[swarmSize][][][];

            double[] particleBestError = new double[swarmSize];

            for (int i = 0; i < swarmSize; i++) //Sets the best Error for each particle to max value.
            {
                particleBestError[i] = double.MaxValue;
            }

            Random rnd = new Random();
            for (int particleNum = 0; particleNum < swarmSize; particleNum++)
            {
                personalBestError.Add(double.MaxValue); //Sets best Errors to be max Value so any Error following is considered an improvement
                currentError.Add(double.MaxValue);
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
                bestInformantErrror.Add(double.MaxValue);
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
                        velocity[particleNum].AddRange(ListMult(rnd.NextDouble(), swarmPositions[particleNum])); //give particle a random velocity created by multiplying its position by a random value.
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
        /// Processes the Input Data File into suitable inputs for the ANN and the ANNs ideal output.
        /// </summary>
        /// <param name="annStructure"> The Structure of the ANN. Represented by an array with the number of nodes in each layer</param>
        private static void DataInputSort(int[] annStructure)
        {
            if (annStructure == null) throw new ArgumentNullException(nameof(annStructure));

            string inputText = System.IO.File.ReadAllText(path);
            string reformattedInputText = Regex.Replace(inputText, @"\s+", " ");   //Fixes formatting of File to make all whitespace " "
            string[] inputSplit = reformattedInputText.Split(' '); 

            int numCount = 0;

            foreach (string value in inputSplit)
            {

                if (value != string.Empty)
                {
                    //Determines if value is an input or output by the number of inputs for the ANN (annStructure[0]) size.
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
        private static void UpdateParticle(double inertiaWeight, double cognitiveWeight, double socialWeight,
            double globalWeight, int[] annStructure, int particleNumber, int numberOfInformantGroups)
        {
            double stepSize = 1;
            var informantGroup = particleNumber % numberOfInformantGroups;

            biases[particleNumber] = new double[annStructure.Length - 1];
            var rnd = new Random();

            double b = rnd.NextDouble() * cognitiveWeight;
            double c = rnd.NextDouble() * socialWeight;
            double d = rnd.NextDouble() * globalWeight;

            velocityChange[particleNumber].Clear();
            for (int x = 0; x < swarmPositions[particleNumber].Count - 1; x++)
            {
                //Debug.WriteLine(inertiaWeight + "" + (velocity[particlenum][x]) + "+" + b + "" + (particlebestpositions[particlenum][x] + "-" + swarmpositions[particlenum][x]) + "+" + c + "" + (informantsbestpositions[informantgroup][x] + "-" + swarmpositions[particlenum][x]) + "+" + d +  "" + (particlebestpositions[bestnet][x]));
                velocityChange[particleNumber].Add((inertiaWeight * (velocity[particleNumber][x])) + (b * (particleBestPositions[particleNumber][x] - swarmPositions[particleNumber][x])) + (c * (informantBestPositions[informantGroup][x] - swarmPositions[particleNumber][x])) + (d * (particleBestPositions[bestnet][x] - swarmPositions[particleNumber][x])));
                //Debug.WriteLine(velocitychange[particlenum][x]);
                velocity[particleNumber][x] = velocityChange[particleNumber][x];
                //Debug.WriteLine(velocity[particlenum][x]);
            }


            for (int x = 0; x < swarmPositions[particleNumber].Count - 1; x++)
            {
                //Debug.WriteLine(swarmpositions[particlenum][x]);
                swarmPositions[particleNumber][x] =
                    (swarmPositions[particleNumber][x] + (stepSize * velocity[particleNumber][x]));
                //Debug.WriteLine(swarmpositions[particlenum][x]);
            }

            int particleposlength;

            particleposlength = swarmPositions[particleNumber].Count;


            int weightpos = 0;
            weights[particleNumber] = new double[annStructure.Length - 1][][];

            for (int layernum = 0; layernum < annStructure.Length - 1; layernum++)
            {
                weights[particleNumber][layernum] = new double[annStructure[layernum]][];

                for (int neuron = 0; neuron < annStructure[layernum]; neuron++)
                {
                    weights[particleNumber][layernum][neuron] = new double[annStructure[layernum + 1]];

                }

            }


            for (int layernum = 0; layernum < annStructure.Length - 1; layernum++)
            {
                //Debug.WriteLine(swarmpositions[particlenum][particleposlength - annStructure.Length - 1 + layernum]);
                swarmPositions[particleNumber][particleposlength - annStructure.Length - 1 + layernum] =
                    (swarmPositions[particleNumber][particleposlength - annStructure.Length - 1 + layernum] > 10D)
                        ? 10D
                        : swarmPositions[particleNumber][particleposlength - annStructure.Length - 1 + layernum];
                swarmPositions[particleNumber][particleposlength - annStructure.Length - 1 + layernum] =
                    (swarmPositions[particleNumber][particleposlength - annStructure.Length - 1 + layernum] < -10D)
                        ? -10D
                        : swarmPositions[particleNumber][particleposlength - annStructure.Length - 1 + layernum];

                biases[particleNumber][layernum] =
                    swarmPositions[particleNumber][particleposlength - annStructure.Length - 1 + layernum];

                for (int neuron = 0; neuron < annStructure[layernum]; neuron++)
                {


                    for (int neuronconnection = 0; neuronconnection < annStructure[layernum + 1]; neuronconnection++)
                    {

                        //swarmpositions[particlenum][weightpos] = (swarmpositions[particlenum][weightpos] > 10D) ? 10D : swarmpositions[particlenum][weightpos];
                        //swarmpositions[particlenum][weightpos] = (swarmpositions[particlenum][weightpos] < -10D) ? -10D : swarmpositions[particlenum][weightpos];

                        weights[particleNumber][layernum][neuron][neuronconnection] =
                            swarmPositions[particleNumber][weightpos];
                        weightpos++;
                    }
                }
            }
        }


        static List<double> ListMult(double i, List<double> list1)
        {
            List<double> newlist = new List<double>();

            for (int x = 0; x < list1.Count; x++)
            {
                newlist.Add((list1[x] * i));
            }

            return newlist;
        }


        /// <summary>
        /// Assesses the Fitness of the Particle based on the output given and the correct output as found in the text file.
        /// </summary>
        /// <param name="output">Output from the ANN</param>
        /// <param name="particlenum">The number of the particle/ANN being assessed.</param>
        /// <param name="epoch">Which maxGenerations of ANN this is. </param>
        /// <returns></returns>
        /// 
        static double AssessFitness(List<double> output, int particlenum, int epoch, int numberOfInformantGroups)
        {
            double Error = 0;
            int informantgroup = particlenum % numberOfInformantGroups;
            currentError[particlenum] = Error;

            for (int x = 0; x < output.Count && x < correctOutput.Count; x++)
            {
                Error = Math.Pow((correctOutput[x] - output[x]), 2);

                currentError[particlenum] += Error;

            }

            //Debug.WriteLine(Error);
            //Debug.WriteLine(correctoutput[0]);
            //Debug.WriteLine(output[0]);

            //Debug.WriteLine(currentError[particlenum] + "   particlenum:" + particlenum);

            if (currentError[particlenum] < bestError)
            {
                bestError = currentError[particlenum];
                bestnet = particlenum;

                Debug.WriteLine("besterror: " + bestError);
                Debug.WriteLine(epoch);
            }

            if (currentError[particlenum] < personalBestError[particlenum])
            {
                //Debug.WriteLine("NEWPB");
                personalBestError[particlenum] = currentError[particlenum];
                particleBestPositions[particlenum].Clear();
                particleBestPositions[particlenum].AddRange(swarmPositions[particlenum]);
                //Debug.WriteLine(pbestError[particlenum]+""+particlenum);

            }

            //Debug.WriteLine(informantgroup);
            //Debug.WriteLine(bestinformanterror[informantgroup]);
            if (currentError[particlenum] < bestInformantErrror[informantgroup])
            {
                //Debug.WriteLine("newbestinformant");
                bestInformantErrror[informantgroup] = currentError[particlenum];
                informantBestPositions[informantgroup].Clear();
                informantBestPositions[informantgroup].AddRange(swarmPositions[particlenum]);
            }
            return bestError / output.Count;

        }
    }
}
