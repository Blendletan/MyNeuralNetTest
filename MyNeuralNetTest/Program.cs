namespace MyNeuralNetTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            double m = 3;
            double b = 5;
            var inputs = new List<List<double>>();
            var outputs = new List<List<double>>();
            Random rng = new Random();
            for (double x = 0; x < 10; x += 0.1)
            {
                inputs.Add(new List<double>());
                inputs.Last().Add(x);
                outputs.Add(new List<double>());
                double y = m*x+b+0.1*rng.NextDouble();
                outputs.Last().Add(y);
            }
            var data = new TrainingData(inputs, outputs);
            Network n = new Network(data, 1, 10);
            int numberOfEpochs = 10000;
            n.Train(numberOfEpochs);
            Console.WriteLine("Weights:");
            var weights = n.GetWeights();
            for (int i = 0; i < weights.Count; i++)
            {
                Console.WriteLine($"Layer {i}:");
                foreach (var v in weights[i])
                {
                    Console.WriteLine(v);
                }
            }
            double testX = 100 * rng.NextDouble();
            double[] testInput = new double[] { testX, 1 };
            double testY = m * testX + b;
            Console.WriteLine($"Testing on input {testX}");
            Console.WriteLine($"ExpectedResult is {testY}");
            var testOutput = n.GetOutputs(testInput);
            Console.WriteLine($"Neural net predicted {testOutput[0]}");
            double error = Math.Abs(testY - testOutput[0]);
            Console.WriteLine($"Error was {error}");
        }
    }
    internal class PRNG
    {
        Random rng;
        public PRNG()
        {
            rng = new Random();
        }
        public double Uniform(double a, double b)
        {
            double unitRandom = rng.NextDouble();
            double output = (b - a) * unitRandom + a;
            return output;
        }
    }
    internal class Node
    {
        readonly bool isInitial;
        double? inputValue;
        public List<Edge>? IncomingEdges { get; private set; }
        readonly Guid id;
        public Node(double input)
        {
            isInitial = true;
            inputValue = input;
            IncomingEdges = null;
            id = Guid.NewGuid();
        }
        public Node(bool b)
        {
            isInitial = b;
            if (isInitial)
            {
                IncomingEdges = null;
                inputValue = 0;
            }
            else
            {
                inputValue = null;
                IncomingEdges = new List<Edge>();
            }
            id = Guid.NewGuid();
        }
        public void UpdateInput(double input)
        {
            if (isInitial == false)
            {
                throw new Exception("Can only push inputs on an initial neuron");
            }
            inputValue = input;
        }
        public void AddInput(Edge e)
        {
            if (isInitial || IncomingEdges == null)
            {
                throw new Exception($"Cannot add input to a initial neuron at neuron {id}");
            }
            IncomingEdges.Add(e);
        }
        public double GetOutput()
        {
            if (isInitial)
            {
                if (inputValue == null)
                {
                    throw new Exception($"Impoperly initialized input neuron {id}, does not contain input value");
                }
                return ActivationFunction(inputValue.Value);
            }
            if (IncomingEdges == null)
            {
                throw new Exception($"Impoperly initialized neuron {id}, is not intial but has no input neurons");
            }
            double output = 0;
            foreach (var v in IncomingEdges)
            {
                output += v.input.GetOutput() * v.Weight;
            }
            return output;
        }
        private double ActivationFunction(double input)
        {
            return Math.Max(0,input);
        }
    }
    internal class Edge
    {
        public double Weight { get; private set; }
        public double PreviousWeight { get; private set; }
        public double PreviousErrorDelta { get; set; }
        public Node input;
        public Node output;
        public Edge(Node inNode, Node outNode)
        {
            input = inNode;
            output = outNode;
            Weight = 1;
        }
        public Edge(Node inNode, Node outNode, double w)
        {
            input = inNode;
            output = outNode;
            Weight = w;
        }
        public void UpdateWeight(double newWeight)
        {
            PreviousWeight = Weight;
            Weight = newWeight;
        }
    }
    internal class Layer
    {
        public List<Node> neurons;
        public Layer(int numberOfNeurons, bool isInitial)
        {
            neurons = new List<Node>();
            for (int i = 0; i < numberOfNeurons; i++)
            {
                neurons.Add(new Node(isInitial));
            }
        }
        public List<Edge> GetIncomingEdges()
        {
            var output = new List<Edge>();
            foreach (var n in neurons)
            {
                foreach (var incoming in n.IncomingEdges)
                {
                    output.Add(incoming);
                }
            }
            return output;
        }
        public static void ConnectLayers(Layer inLayer, Layer outLayer)
        {
            foreach (var input in inLayer.neurons)
            {
                foreach (var output in outLayer.neurons)
                {
                    var nextEdge = new Edge(input, output);
                    output.AddInput(nextEdge);
                }
            }
        }
    }
    internal class Network
    {
        List<Layer> layers;
        TrainingData data;
        const double learningRate = 0.00001;
        const double smallNumber = 0.00001;
        PRNG pRng;
        public Network(TrainingData d, int numberOfHiddenLayers, int sizeOfHiddenLayers)
        {
            pRng = new PRNG();
            data = d;
            layers = new List<Layer>();
            int numberOfInputs = d.NumberOfInputVariables;
            var inputLayer = new Layer(numberOfInputs, true);
            layers.Add(inputLayer);
            for (int i = 0; i < numberOfHiddenLayers; i++)
            {
                var nextLayer = new Layer(sizeOfHiddenLayers, false);
                Layer.ConnectLayers(layers.Last(), nextLayer);
                layers.Add(nextLayer);
            }
            var outputLayer = new Layer(d.NumberOfOutputVariables, false);
            Layer.ConnectLayers(layers.Last(), outputLayer);
            layers.Add(outputLayer);
        }
        public List<List<double>> GetWeights()
        {
            var output = new List<List<double>>();
            for (int i = 1; i < layers.Count; i++)
            {
                output.Add(new List<double>());
                foreach (var v in layers[i].GetIncomingEdges())
                {
                    output.Last().Add(v.Weight);
                }
            }
            return output;
        }
        public double GetTotalError()
        {
            double totalError = 0;
            for (int i = 0; i < data.NumberOfSamples; i++)
            {
                var nextOutput = GetOutputs(data.GetInputs(i));
                var nextError = GetError(nextOutput, i);
                totalError += nextError;
            }
            return totalError;
        }
        public double GetError(double[] outputs, int sampleIndex)
        {
            if (outputs.Length != data.NumberOfOutputVariables)
            {
                throw new Exception("invalid trial outputs");
            }
            var expectedOutputs = data.GetOutputs(sampleIndex);
            double totalSquareError = 0;
            for (int i = 0; i < outputs.Length; i++)
            {
                double error = outputs[i] - expectedOutputs[i];
                double squareError = error * error;
                totalSquareError += squareError;
            }
            return totalSquareError;
        }
        public void Train(int numberOfEpochs)
        {
            InitialTrainingEpoch();
            for (int i = 0; i < numberOfEpochs; i++)
            {
                TrainingIteration();
            }
        }
        private void InitialTrainingEpoch()
        {
            for (int i = 1; i < layers.Count; i++)
            {
                foreach (var synapse in layers[i].GetIncomingEdges())
                {
                    double previousError = GetTotalError();
                    double weightOffset = pRng.Uniform(-smallNumber, smallNumber);
                    double newWeight = synapse.Weight + weightOffset;
                    synapse.UpdateWeight(newWeight);
                    double newError = GetTotalError();
                    double errorDelta = newError - previousError;
                    synapse.PreviousErrorDelta = errorDelta;
                }
            }
        }
        private void TrainingIteration()
        {
            for (int i = 1; i < layers.Count; i++)
            {
                foreach (var synapse in layers[i].GetIncomingEdges())
                {
                    double previousError = GetTotalError();
                    double weightDelta = synapse.Weight - synapse.PreviousWeight;
                    double previousErrorDelta = synapse.PreviousErrorDelta;
                    double newWeight = synapse.Weight;
                    if (weightDelta == 0 || previousErrorDelta == 0)
                    {
                        newWeight += pRng.Uniform(-smallNumber, smallNumber);
                    }
                    else
                    {
                        double partialDerivative = previousErrorDelta / weightDelta;
                        newWeight -= learningRate * partialDerivative;
                    }
                    synapse.UpdateWeight(newWeight);
                    double newError = GetTotalError();
                    double errorDelta = newError - previousError;
                    synapse.PreviousErrorDelta = errorDelta;
                }
            }
        }
        public double[] GetOutputs()
        {
            var outputLayer = layers.Last();
            List<double> output = new List<double>();
            foreach (var v in outputLayer.neurons)
            {
                output.Add(v.GetOutput());
            }
            return output.ToArray();
        }
        public double[] GetOutputs(double[] inputs)
        {
            UpdateInputs(inputs);
            return GetOutputs();
        }
        public void UpdateInputs(double[] inputs)
        {
            if (layers.First().neurons.Count != inputs.Length)
            {
                throw new Exception("Invalid number of input variables");
            }
            for (int i = 0; i < inputs.Length; i++)
            {
                layers.First().neurons[i].UpdateInput(inputs[i]);
            }
        }
    }
    internal class TrainingData
    {
        public readonly int NumberOfSamples;
        public readonly int NumberOfInputVariables;
        public readonly int NumberOfOutputVariables;
        public readonly double[,] inputData;
        public readonly double[,] outputData;
        public double[] GetInputs(int sampleIndex)
        {
            var output = new double[NumberOfInputVariables];
            for (int i = 0; i < NumberOfInputVariables; i++)
            {
                output[i] = inputData[sampleIndex, i];
            }
            return output;
        }
        public double[] GetOutputs(int sampleIndex)
        {
            var output = new double[NumberOfOutputVariables];
            for (int i = 0; i < NumberOfOutputVariables; i++)
            {
                output[i] = outputData[sampleIndex, i];
            }
            return output;
        }
        public TrainingData(List<List<double>> inputs, List<List<double>> outputs)
        {
            if (inputs.Count != outputs.Count)
            {
                throw new Exception("Number of inputs does not match number of outputs in training data");
            }
            NumberOfSamples = inputs.Count;
            NumberOfInputVariables = inputs[0].Count + 1;
            NumberOfOutputVariables = outputs[0].Count;
            inputData = new double[NumberOfSamples, NumberOfInputVariables];
            outputData = new double[NumberOfSamples, NumberOfOutputVariables];
            for (int i = 0; i < NumberOfSamples; i++)
            {
                if (inputs[i].Count != NumberOfInputVariables - 1)
                {
                    throw new Exception($"Missing training data, wrong number of inputs in row {i}");
                }
                if (outputs[i].Count != NumberOfOutputVariables)
                {
                    throw new Exception($"Missing training data, wrong number of outputs in row {i}");
                }
                for (int j = 0; j < NumberOfInputVariables - 1; j++)
                {
                    inputData[i, j] = inputs[i][j];
                }
                inputData[i, NumberOfInputVariables - 1] = 1;
                for (int j = 0; j < NumberOfOutputVariables; j++)
                {
                    outputData[i, j] = outputs[i][j];
                }
            }
        }
    }
}
