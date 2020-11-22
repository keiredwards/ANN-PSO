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
        private const int V = 0;

        static double[][][] particleBestlocation;
        public static double bestError = double.MaxValue; //Sets best Error to max value
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

        public PSO(int swarmsize, int[] ANNStructure, int epoch, int activationfunction, double inertiaweight, double cognitiveweight, double socialweight, double globalweight, int numberOfInformantGroups)
        {


            string inputtext = System.IO.File.ReadAllText(path);

            string withoutwhitespace = Regex.Replace(inputtext, @"\s+", " ");


            string[] inputsplit = withoutwhitespace.Split(' ');

            int numcount = 0;

            foreach (string num in inputsplit)
            {

                if (num != string.Empty)
                {
                    //Debug.WriteLine(ANNStructure[0]);
                    if (numcount <= ANNStructure[0] - 1)
                    {
                        //Debug.WriteLine("yes");
                        //Debug.WriteLine((num));
                        input.Add(Convert.ToDouble(num));

                        numcount++;
                    }
                    else
                    {
                        correctOutput.Add(Convert.ToDouble(num));
                        //Debug.WriteLine(Convert.ToDouble(num));
                        numcount = 0;
                    }

                }



            }



            bestLocations = new double[ANNStructure.Length][];

            particleBestlocation = new double[swarmsize][][];
            biases = new double[swarmsize][];
            weights = new double[swarmsize][][][];
            //Debug.WriteLine("oops");



            double[] particleBestError = new double[swarmsize];

            for (int i = 0; i < swarmsize; i++)       //Sets the best Error for each particle to max value.
            {
                particleBestError[i] = double.MaxValue;
            }

            Random rnd = new Random();
            for (int particlenum = 0; particlenum < swarmsize; particlenum++)              //Give Random Init Velocities and Positions to Particles
            {


                personalBestError.Add(double.MaxValue);
                currentError.Add(double.MaxValue);
                swarmPositions.Add(new List<double>());
                newSwarmPositions.Add(new List<double>());
                informantBestPositions.Add(new List<double>());
                particleBestPositions.Add(new List<double>());
                velocity.Add(new List<double>());
                velocityChange.Add(new List<double>());
                output.Add(new List<double>());
                //SETS VELOCITY TO BE THE SAME AS RANDOM POSITION
            }


            for (int i = 0; i < 5; i++)
            {
                bestInformantErrror.Add(double.MaxValue);
            }


            int x = 0;
            while (x < epoch)
            {
                for (int particlenum = 0; particlenum < swarmsize; particlenum++)
                {

                    if (x == 0)
                    {
                        new ANN(particlenum, null, null, activationfunction, ANNStructure);
                        velocity[particlenum].Clear();
                        velocity[particlenum].AddRange(ListMult(rnd.NextDouble(), swarmPositions[particlenum]));
                        particleBestPositions[particlenum].AddRange(swarmPositions[particlenum]);
                        //Debug.WriteLine(ANN.weights[particlenum][0][0]);

                    }
                    else
                    {
                        swarmPositions[particlenum].Clear();
                        new ANN(particlenum, weights[particlenum], biases[particlenum], activationfunction, ANNStructure);

                    }
                }
                
                for (int particlenum = 0; particlenum < swarmsize; particlenum++)
                {
                    AssessFitness(output[particlenum], particlenum, x);
                    double temp = 0.5 / 1000;
                    if (inertiaweight > 0.4) inertiaweight -= temp;
                    UpdateParticle(inertiaweight, cognitiveweight, socialweight, globalweight, ANNStructure, particlenum, numberOfInformantGroups);
                }
                x++;
            }

        }

        /// <summary>
        /// Updates Particle to find its new position based on its current velocity,current error, its best error, its informants best errors and the global best error.
        /// </summary>
        /// <param name="inertiaweight">How much of the velocity of the particle is carried onto its next iteration</param>
        /// <param name="cognitiveweight">The pull of the particle towards its best ever position</param>
        /// <param name="socialweight">The pull of the particle towards its informant's best ever position</param>
        /// <param name="globalweight">The pull of the particle towards the global best ever position</param>
        /// <param name="annStructure">The Layout of the ANN</param>
        /// <param name="particleNumber">The ANN's index.</param>
        static void UpdateParticle(double inertiaweight, double cognitiveweight, double socialweight, double globalweight, int[] annStructure, int particleNumber, int numberOfInformantGroups)
        {
            //Debug.WriteLine(swarmpositions[particlenum].Count);
            int informantgroup;
            double stepsize = 1;
            informantgroup = particleNumber % numberOfInformantGroups;

            biases[particleNumber] = new double[annStructure.Length - 1];
            Random rnd = new Random();

            double b = rnd.NextDouble() * cognitiveweight;
            double c = rnd.NextDouble() * socialweight;
            double d = rnd.NextDouble() * globalweight;



            velocityChange[particleNumber].Clear();
            for (int x = 0; x < swarmPositions[particleNumber].Count - 1; x++)
            {
                //Debug.WriteLine(inertiaweight + "*" + (velocity[particlenum][x]) + "+" + b + "*" + (particlebestpositions[particlenum][x] + "-" + swarmpositions[particlenum][x]) + "+" + c + "*" + (informantsbestpositions[informantgroup][x] + "-" + swarmpositions[particlenum][x]) + "+" + d +  "*" + (particlebestpositions[bestnet][x]));
                velocityChange[particleNumber].Add((inertiaweight * (velocity[particleNumber][x])) + (b * (particleBestPositions[particleNumber][x] - swarmPositions[particleNumber][x])) + (c * (informantBestPositions[informantgroup][x] - swarmPositions[particleNumber][x])) + (d * (particleBestPositions[bestnet][x] - swarmPositions[particleNumber][x])));
                //Debug.WriteLine(velocitychange[particlenum][x]);
                velocity[particleNumber][x] = velocityChange[particleNumber][x];
                //Debug.WriteLine(velocity[particlenum][x]);
            }


            for (int x = 0; x < swarmPositions[particleNumber].Count - 1; x++)
            {
                //Debug.WriteLine(swarmpositions[particlenum][x]);
                swarmPositions[particleNumber][x] = (swarmPositions[particleNumber][x] + (stepsize * velocity[particleNumber][x]));
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
                //Debug.WriteLine(swarmpositions[particlenum][particleposlength - ANNStructure.Length - 1 + layernum]);
                swarmPositions[particleNumber][particleposlength - annStructure.Length - 1 + layernum] = (swarmPositions[particleNumber][particleposlength - annStructure.Length - 1 + layernum] > 10D) ? 10D : swarmPositions[particleNumber][particleposlength - annStructure.Length - 1 + layernum];
                swarmPositions[particleNumber][particleposlength - annStructure.Length - 1 + layernum] = (swarmPositions[particleNumber][particleposlength - annStructure.Length - 1 + layernum] < -10D) ? -10D : swarmPositions[particleNumber][particleposlength - annStructure.Length - 1 + layernum];

                biases[particleNumber][layernum] = swarmPositions[particleNumber][particleposlength - annStructure.Length - 1 + layernum];

                for (int neuron = 0; neuron < annStructure[layernum]; neuron++)
                {


                    for (int neuronconnection = 0; neuronconnection < annStructure[layernum + 1]; neuronconnection++)
                    {

                        //swarmpositions[particlenum][weightpos] = (swarmpositions[particlenum][weightpos] > 10D) ? 10D : swarmpositions[particlenum][weightpos];
                        //swarmpositions[particlenum][weightpos] = (swarmpositions[particlenum][weightpos] < -10D) ? -10D : swarmpositions[particlenum][weightpos];

                        weights[particleNumber][layernum][neuron][neuronconnection] = swarmPositions[particleNumber][weightpos];
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
        /// <param name="epoch">Which epoch of ANN this is. </param>
        /// <returns></returns>
        /// 
        static double AssessFitness(List<double> output, int particlenum, int epoch) 
        {
            double Error = 0;
            int informantgroup = particlenum % 5;
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
            //Debug.WriteLine(particlenum);
            //Debug.WriteLine("error:" + Error);

            //Debug.WriteLine("besterror: " + bestError);
            //Debug.WriteLine("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            //Debug.WriteLine(bestError / output.Count + "   particlenum:" + particlenum);
            return bestError / output.Count;

        }
    }
}
