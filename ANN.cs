using System;
using System.Diagnostics;

namespace ANN_PSO
{
    class Ann
    {
        private readonly int _maxNeuronsInHiddenLayers;
        private static double[][] _values;
        private static int _inputLayerSize;

        public Ann(int particleNum, double[][][] weights, double[] biases, int activationFunction, int[] annStructure)
        {
            _inputLayerSize = annStructure[0];
            int outputLayerSize = annStructure[^1];

            foreach (int layerSize in annStructure)
            {
                if (layerSize > _maxNeuronsInHiddenLayers)
                {
                    _maxNeuronsInHiddenLayers = layerSize;
                }
            }

            int hiddenLayerCount = annStructure.Length - 2;
            _values = new double[annStructure.Length][];

            for (int layerNum = 0; layerNum < annStructure.Length; layerNum++)
            {
                _values[layerNum] = new double[annStructure[layerNum]];

                for (int neuron = 0; neuron < annStructure[layerNum]; neuron++)
                {
                    //sets Values of each Neuron to 0
                    _values[layerNum][neuron] = 0;  
                }
            }

            Random rnd = new Random();

            //Generate Random Weights for every Neuron Connection
            if (weights == null) 
            {
                weights = new double[hiddenLayerCount + 1][][];

                for (int layerNum = 0; layerNum < hiddenLayerCount + 1; layerNum++)                    
                {
                    weights[layerNum] = new double[annStructure[layerNum]][];

                    for (int neuron = 0; neuron < annStructure[layerNum]; neuron++) 
                    {
                        weights[layerNum][neuron] = new double[annStructure[layerNum + 1]];

                        for (int neuronConnection = 0; neuronConnection < annStructure[layerNum + 1]; neuronConnection++) 
                        {
                            weights[layerNum][neuron][neuronConnection] = rnd.NextDouble();
                        }
                    }
                }
            }

            //Generate Biases for every Layer
            if (biases == null) 
            {
                biases = new double[hiddenLayerCount + 1];
                for (int layerNum = 0; layerNum < hiddenLayerCount + 1; layerNum++)
                {
                    biases[layerNum] = rnd.NextDouble();
                }
            }

            Pso.Output[particleNum].Clear();
            double[] inputX = new double[_inputLayerSize];

            //Current Implementation only works for 1 or 2 inputs to a single output as required for coursework.
            if (_inputLayerSize == 1)    
            {
                for (int inputNumber = 0; inputNumber < Pso.Input.Count; inputNumber++)
                {
                    inputX[0] = Pso.Input[inputNumber];
                    Run(this, _maxNeuronsInHiddenLayers, annStructure, outputLayerSize, particleNum, weights, biases, inputX, inputNumber, activationFunction);
                }
            }
            else if (_inputLayerSize == 2)
            {

                for (int inputNumber = 0; inputNumber < Pso.Input.Count / 2; inputNumber += 2)
                {
                    inputX[0] = Pso.Input[inputNumber];
                    inputX[1] = Pso.Input[inputNumber + 1];
                    Run(this, _maxNeuronsInHiddenLayers, annStructure, outputLayerSize, particleNum, weights, biases, inputX, inputNumber, activationFunction);
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
        /// <param name="maxNeuronsInHiddenLayers">Maximum Hidden Layer Size</param>
        /// <param name="annStructure">The Sizes of Each of the ANNs Layers</param>
        /// <param name="outputLayerSize">The Number of Outputs</param>
        /// <param name="particleNum">The ANN's identifying particle number in the Particle Swarm</param>
        /// <param name="weights">Weights of Each Neuron in the ANN</param>
        /// <param name="biases">Biases of Each Layer in the ANN</param>
        /// <param name="input">Input from the Data Set</param>
        /// <param name="inputNumber">Function is called for Each input, denoted which of the inputs this is</param>
        /// <param name="activationFunction">The Activation function applied to the ANN</param>
        public static void Run(Ann ann, int maxNeuronsInHiddenLayers, int[] annStructure, int outputLayerSize, int particleNum, double[][][] weights, double[] biases, double[] input, int inputNumber, int activationFunction)
        {
            //clears values
            for (int layerNum = 0; layerNum < annStructure.Length; layerNum++)     
            {
                for (int neuron = 0; neuron < annStructure[layerNum]; neuron++)
                {
                    _values[layerNum][neuron] = 0;

                }
            }

            //Sets Input Values
            _values[0][0] = input[0]; 

            if (_inputLayerSize == 2)
            {
                _values[0][1] = input[1];
            }

            for (int layerNum = 0; layerNum < annStructure.Length - 1; layerNum++) 
            {
                for (int neuron = 0; neuron < annStructure[layerNum]; neuron++)
                {
                    //to be improved
                    if (neuron == 0) 
                    {
                        _values[layerNum][neuron] += biases[layerNum];
                    }

                    for (int neuronConnection = 0; neuronConnection < annStructure[layerNum + 1]; neuronConnection++)
                    {
                        //Calculates Values of next layer based on Neuron value and connection weight
                        _values[layerNum + 1][neuronConnection] += _values[layerNum][neuron] * weights[layerNum][neuron][neuronConnection];   
                    }
                }

            }

            //Applies Activation Function on Neuron Value
            for (int layerNum = 1; layerNum < annStructure.Length; layerNum++) 
            {
                for (int neuron = 0; neuron < annStructure[layerNum]; neuron++)
                {
                    _values[layerNum][neuron] = ActivationFunction(_values[layerNum][neuron], activationFunction);
                }
            }

            //Sets output value
            Pso.Output[particleNum].Add(_values[annStructure.Length - 1][0]);

            //Add Weights
            //Only adds weights if its the first time the ANN is generated
            if (inputNumber == 0)  
            {
                for (int layerNum = 0; layerNum < annStructure.Length - 1; layerNum++) 
                {
                    for (int neuron = 0; neuron < annStructure[layerNum]; neuron++)
                    {
                        for (int neuronConnection = 0; neuronConnection < annStructure[layerNum + 1]; neuronConnection++)
                        {
                            Pso.SwarmPositions[particleNum].Add(weights[layerNum][neuron][neuronConnection]);
                        }
                    }
                }

                //Adds Biases
                for (int layerNum = 0; layerNum < annStructure.Length - 1; layerNum++)      
                {
                    Pso.SwarmPositions[particleNum].Add(biases[layerNum]);
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



