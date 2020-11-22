using System;
using System.Collections.Generic;
using System.Reflection.Metadata;
using System.Text;
using System.IO;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Numerics;

namespace ANN_PSO
{
    class ANN
    {


        int maxNeuronsInHiddenLayers;
        int outputLayerSize;
        static double[][] values;
        public static int inputLayerSize;



        public ANN(int particleNum, double[][][] weights, double[] biases, int activationFunction, int[] ANNStructure)
        {

            for (int layer = 0; layer < ANNStructure.Length; layer++)
            {
                if (layer == 0)
                {
                    inputLayerSize = ANNStructure[layer];
                    //Debug.WriteLine(ANNStructure[layer]);
                }
                else if (layer == ANNStructure.Length)
                {
                    outputLayerSize = ANNStructure[layer];
                }
                else
                {
                    if (ANNStructure[layer] > maxNeuronsInHiddenLayers)
                    {
                        maxNeuronsInHiddenLayers = ANNStructure[layer];
                    }
                }
            }


            int HiddenLayerCount = ANNStructure.Length - 2;


            values = new double[ANNStructure.Length][];

            for (int LayerNum = 0; LayerNum < ANNStructure.Length; LayerNum++)
            {
                values[LayerNum] = new double[ANNStructure[LayerNum]];

                for (int Neuron = 0; Neuron < ANNStructure[LayerNum]; Neuron++)
                {

                    values[LayerNum][Neuron] = 0;
                }
            }
            //Debug.WriteLine(values[LayerNum][Neuron]);
            Random rnd = new Random();

            if (weights == null)
            {

                weights = new double[HiddenLayerCount + 1][][];
                //Debug.WriteLine("weights were null");
                for (int LayerNum = 0; LayerNum < HiddenLayerCount + 1; LayerNum++) //Layer Loop                      //Generate Random Weights for every 
                {
                    weights[LayerNum] = new double[ANNStructure[LayerNum]][];


                    //Debug.WriteLine(LayerNum);
                    for (int Neuron = 0; Neuron < ANNStructure[LayerNum]; Neuron++) //Neuron Loop
                    {
                        weights[LayerNum][Neuron] = new double[ANNStructure[LayerNum + 1]];
                        for (int NeuronConnection = 0; NeuronConnection < ANNStructure[LayerNum + 1]; NeuronConnection++) //Neuron Connection Loop
                        {

                            double NeuronWeight = rnd.NextDouble();

                            weights[LayerNum][Neuron][NeuronConnection] = NeuronWeight;
                            //Debug.WriteLine(NeuronWeight + "hey");
                            //Debug.WriteLine("LayerNum:" + LayerNum + "Neuron:" + Neuron + "NeuronConnection" + NeuronConnection + "NeuronWeight:" + NeuronWeight);

                        }
                    }
                }
            }

            if (biases == null)
            {
                //Debug.WriteLine("biases were null");

                biases = new double[HiddenLayerCount + 1];
                for (int LayerNum = 0; LayerNum < HiddenLayerCount + 1; LayerNum++)
                {
                    biases[LayerNum] = rnd.NextDouble();

                }
            }

            PSO.output[particleNum].Clear();

            double[] inputx = new double[inputLayerSize];
            //Debug.WriteLine(InputLayerSize);
            if (inputLayerSize == 1)
            {
                //Debug.WriteLine(PSO.input.Count);
                for (int inputNumber = 0; inputNumber < PSO.input.Count; inputNumber++)
                {

                    inputx[0] = PSO.input[inputNumber];
                    Run(this, maxNeuronsInHiddenLayers, ANNStructure, outputLayerSize, particleNum, weights, biases, inputx, inputNumber, activationFunction);
                }
            }
            else if (inputLayerSize == 2)
            {

                //Debug.WriteLine(PSO.input.Count);

                for (int inputnumber = 0; inputnumber < PSO.input.Count / 2; inputnumber += 2)
                {


                    inputx[0] = PSO.input[inputnumber];
                    inputx[1] = PSO.input[inputnumber + 1];
                    Run(this, maxNeuronsInHiddenLayers, ANNStructure, outputLayerSize, particleNum, weights, biases, inputx, inputnumber, activationFunction);
                }
            }

        }


        public static void Run(ANN ann, int MaxNeuronsInHiddenLayers, int[] ANNStructure, int OutputLayerSize, int particleNum, double[][][] weights, double[] biases, double[] input, int inputNumber, int activationFunction)
        {




            for (int LayerNum = 0; LayerNum < ANNStructure.Length; LayerNum++)      //Sets all values to 0 - clears values
            {
                for (int Neuron = 0; Neuron < ANNStructure[LayerNum]; Neuron++)
                {
                    values[LayerNum][Neuron] = 0;

                }
            }



            values[0][0] = input[0];  //Sets Input Values
            //Debug.WriteLine(values[0][0]);

            if (inputLayerSize == 2)
            {
                values[0][1] = input[1];
                //Debug.WriteLine(values[0][1]);
            }



            for (int LayerNum = 0; LayerNum < ANNStructure.Length - 1; LayerNum++) // For each Layer
            {
                //Debug.WriteLine("hello?");
                for (int Neuron = 0; Neuron < ANNStructure[LayerNum]; Neuron++) //Neuron Loop
                {
                    if (Neuron == 0) //this does need fixed
                    {
                        values[LayerNum][Neuron] += biases[LayerNum];
                    }

                    for (int NeuronConnection = 0; NeuronConnection < ANNStructure[LayerNum + 1]; NeuronConnection++)
                    {
                        //Debug.WriteLine(LayerNum.ToString() + Neuron.ToString() + NeuronConnection.ToString());
                        //Debug.WriteLine(weights[LayerNum][Neuron][NeuronConnection]);
                        //Debug.WriteLine(values[LayerNum][Neuron]);
                        values[LayerNum + 1][NeuronConnection] += values[LayerNum][Neuron] * weights[LayerNum][Neuron][NeuronConnection];

                        //Debug.WriteLine(weights[LayerNum][Neuron][NeuronConnection]);
                    }
                }

            }


            for (int LayerNum = 1; LayerNum < ANNStructure.Length; LayerNum++)          //APPLY ACTIVATION FUNCTION
            {
                for (int Neuron = 0; Neuron < ANNStructure[LayerNum]; Neuron++)
                {
                    //Debug.WriteLine(values[LayerNum][Neuron]);
                    values[LayerNum][Neuron] = ActivationFunction(values[LayerNum][Neuron], activationFunction);
                    //Debug.WriteLine(values[LayerNum][Neuron]);




                }
            }

            //Debug.WriteLine(values[ANNStructure.Length - 1][0]);
            PSO.output[particleNum].Add(values[ANNStructure.Length - 1][0]); //Sets out


            if (inputNumber == 0)  //we only need to add the weights if its the first time they are generated, no use updating every time with same weights.
            {

                for (int LayerNum = 0; LayerNum < ANNStructure.Length - 1; LayerNum++)  //ADD WEIGHTS TO SWARMPOSITIONS
                {
                    for (int Neuron = 0; Neuron < ANNStructure[LayerNum]; Neuron++)
                    {
                        for (int NeuronConnection = 0; NeuronConnection < ANNStructure[LayerNum + 1]; NeuronConnection++)
                        {

                            PSO.swarmPositions[particleNum].Add(weights[LayerNum][Neuron][NeuronConnection]);

                            //Debug.WriteLine(LayerNum.ToString() + Neuron.ToString() + NeuronConnection.ToString());
                        }
                    }
                }

                for (int LayerNum = 0; LayerNum < ANNStructure.Length - 1; LayerNum++)      //add biases
                {
                    PSO.swarmPositions[particleNum].Add(biases[LayerNum]);
                    //Debug.WriteLine(biases[LayerNum]);
                    //Debug.WriteLine(LayerNum);
                }
            }

        }

        public static double ActivationFunction(double value, int activationFunction)
        {
            //Debug.WriteLine(value);

            switch (activationFunction)
            {
                case 0:     //Null
                    value = 0;
                    //Debug.WriteLine(0);
                    break;

                case 1:     //Sigmoid
                    value = 1 / (1 + Math.Pow(Math.E, -value));
                    //Debug.WriteLine(1);
                    break;

                case 2:     //Hyperbolic Tangent
                    value = Math.Tanh(value);
                    //Debug.WriteLine(2);
                    break;
                case 3:
                    value = Math.Cos(value);
                    //Debug.WriteLine(3);
                    break;
                case 4:
                    value = Math.Exp(-((value * value) / 2));
                    //Debug.WriteLine(4);
                    break;
                case 5:         //non-linear cube
                    value = value * value * value;
                    //Debug.WriteLine(5);
                    break;
            }
            //Debug.WriteLine(value);
            return value;
        }
    }
}



