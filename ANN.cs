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
            inputLayerSize = ANNStructure[0];
            outputLayerSize = ANNStructure[ANNStructure.Length-1];

            foreach (var layerSize in ANNStructure)
            {
                if (layerSize > maxNeuronsInHiddenLayers)
                {
                    maxNeuronsInHiddenLayers = layerSize;
                }
            }

            int HiddenLayerCount = ANNStructure.Length - 2;
            values = new double[ANNStructure.Length][];

            for (int LayerNum = 0; LayerNum < ANNStructure.Length; LayerNum++)
            {
                values[LayerNum] = new double[ANNStructure[LayerNum]];

                for (int Neuron = 0; Neuron < ANNStructure[LayerNum]; Neuron++)
                {
                    //sets Values of each Neuron to 0
                    values[LayerNum][Neuron] = 0;  
                }
            }

            Random rnd = new Random();

            //Generate Random Weights for every Neuron Connection
            if (weights == null) 
            {
                weights = new double[HiddenLayerCount + 1][][];

                for (int LayerNum = 0; LayerNum < HiddenLayerCount + 1; LayerNum++)                    
                {
                    weights[LayerNum] = new double[ANNStructure[LayerNum]][];

                    for (int Neuron = 0; Neuron < ANNStructure[LayerNum]; Neuron++) 
                    {
                        weights[LayerNum][Neuron] = new double[ANNStructure[LayerNum + 1]];

                        for (int NeuronConnection = 0; NeuronConnection < ANNStructure[LayerNum + 1]; NeuronConnection++) 
                        {
                            weights[LayerNum][Neuron][NeuronConnection] = rnd.NextDouble();
                        }
                    }
                }
            }

            //Generate Biases for every Layer
            if (biases == null) 
            {
                biases = new double[HiddenLayerCount + 1];
                for (int LayerNum = 0; LayerNum < HiddenLayerCount + 1; LayerNum++)
                {
                    biases[LayerNum] = rnd.NextDouble();
                }
            }

            PSO.Output[particleNum].Clear();
            double[] inputx = new double[inputLayerSize];

            //Current Implementation only works for 1 or 2 inputs to a single output as required for coursework.
            if (inputLayerSize == 1)    
            {
                for (int inputNumber = 0; inputNumber < PSO.Input.Count; inputNumber++)
                {
                    inputx[0] = PSO.Input[inputNumber];
                    Run(this, maxNeuronsInHiddenLayers, ANNStructure, outputLayerSize, particleNum, weights, biases, inputx, inputNumber, activationFunction);
                }
            }
            else if (inputLayerSize == 2)
            {

                for (int inputnumber = 0; inputnumber < PSO.Input.Count / 2; inputnumber += 2)
                {
                    inputx[0] = PSO.Input[inputnumber];
                    inputx[1] = PSO.Input[inputnumber + 1];
                    Run(this, maxNeuronsInHiddenLayers, ANNStructure, outputLayerSize, particleNum, weights, biases, inputx, inputnumber, activationFunction);
                }
            }
            else
            {
                Debug.Write("Current Implementation only works for 1 or 2 inputs to a single output as required for coursework");
            }

        }

        /// <summary>
        /// Creates ANN and processes Output using provided parameters.
        /// </summary>
        /// <param name="ann">The ANN</param>
        /// <param name="MaxNeuronsInHiddenLayers">Maximum Hidden Layer Size</param>
        /// <param name="ANNStructure">The Sizes of Each of the ANNs Layers</param>
        /// <param name="OutputLayerSize">The Number of Outputs</param>
        /// <param name="particleNum">The ANN's identifying particle number in the Particle Swarm</param>
        /// <param name="weights">Weights of Each Neuron in the ANN</param>
        /// <param name="biases">Biases of Each Layer in the ANN</param>
        /// <param name="input">Input from the Data Set</param>
        /// <param name="inputNumber">Function is called for Each input, denoted which of the inputs this is</param>
        /// <param name="activationFunction">The Activation function applied to the ANN</param>
        public static void Run(ANN ann, int MaxNeuronsInHiddenLayers, int[] ANNStructure, int OutputLayerSize, int particleNum, double[][][] weights, double[] biases, double[] input, int inputNumber, int activationFunction)
        {
            //clears values
            for (int LayerNum = 0; LayerNum < ANNStructure.Length; LayerNum++)     
            {
                for (int Neuron = 0; Neuron < ANNStructure[LayerNum]; Neuron++)
                {
                    values[LayerNum][Neuron] = 0;

                }
            }

            //Sets Input Values
            values[0][0] = input[0]; 

            if (inputLayerSize == 2)
            {
                values[0][1] = input[1];
            }

            for (int LayerNum = 0; LayerNum < ANNStructure.Length - 1; LayerNum++) 
            {
                for (int Neuron = 0; Neuron < ANNStructure[LayerNum]; Neuron++)
                {
                    //to be improved
                    if (Neuron == 0) 
                    {
                        values[LayerNum][Neuron] += biases[LayerNum];
                    }

                    for (int NeuronConnection = 0; NeuronConnection < ANNStructure[LayerNum + 1]; NeuronConnection++)
                    {
                        //Calculates Values of next layer based on Neuron value and connection weight
                        values[LayerNum + 1][NeuronConnection] += values[LayerNum][Neuron] * weights[LayerNum][Neuron][NeuronConnection];   
                    }
                }

            }

            //Applies Activation Function on Neuron Value
            for (int LayerNum = 1; LayerNum < ANNStructure.Length; LayerNum++) 
            {
                for (int Neuron = 0; Neuron < ANNStructure[LayerNum]; Neuron++)
                {
                    values[LayerNum][Neuron] = ActivationFunction(values[LayerNum][Neuron], activationFunction);
                }
            }

            //Sets output value
            PSO.Output[particleNum].Add(values[ANNStructure.Length - 1][0]);

            //Add Weights
            //Only adds weights if its the first time the ANN is generated
            if (inputNumber == 0)  
            {
                for (int LayerNum = 0; LayerNum < ANNStructure.Length - 1; LayerNum++) 
                {
                    for (int Neuron = 0; Neuron < ANNStructure[LayerNum]; Neuron++)
                    {
                        for (int NeuronConnection = 0; NeuronConnection < ANNStructure[LayerNum + 1]; NeuronConnection++)
                        {
                            PSO.swarmPositions[particleNum].Add(weights[LayerNum][Neuron][NeuronConnection]);
                        }
                    }
                }

                //Adds Biases
                for (int LayerNum = 0; LayerNum < ANNStructure.Length - 1; LayerNum++)      
                {
                    PSO.swarmPositions[particleNum].Add(biases[LayerNum]);
                }
            }

        }

        /// <summary>
        /// Processes a value through the specified Activation function
        /// </summary>
        /// <param name="value">The value to be processed</param>
        /// <param name="activationFunction">Integer representing chosen activation function. 0: NULL, 1: Sigmoid , 2:Hyperbolic Tangent, 3: Cosine, 4: Math.Exp(-((value * value) / 2)), 5: non-linear cubic</param>
        /// <returns>Value from activation function</returns>
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



