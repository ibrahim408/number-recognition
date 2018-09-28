import csv
import numpy as np


class NumberRecognition:
    def __init__(self, number_of_input_nodes, number_of_hidden_node, number_of_output_node, learning_rate):
        self.num_input_nodes = number_of_input_nodes
        self.num_hidden_node = number_of_hidden_node
        self.num_output_nodes = number_of_output_node
        self.learning_rate = learning_rate

        i_weight = 1 / pow(self.num_hidden_node, .5)
        i_weight2 = 1 / pow(self.num_output_nodes, .5)
        self.weight_input_to_hidden = np.random.normal(0, i_weight, size=(self.num_hidden_node, self.num_input_nodes))
        self.weight_hidden_to_output = np.random.normal(0, i_weight2, size=(self.num_output_nodes, self.num_hidden_node))
        print(self.weight_hidden_to_output.shape)

    def setTrainingTest(self, training, test):
        self.training_count = training
        self.test_count = test

    def query(self):
        self.X_hidden = np.dot(self.weight_input_to_hidden, self.input)
        self.O_hidden = self.sigmoid(self.X_hidden)

        self.X_output = np.dot(self.weight_hidden_to_output, self.O_hidden)
        self.O_output = self.sigmoid(self.X_output)

    def backPropagating(self):
        self.error_output = self.answer - self.O_output
        self.error_hidden = np.dot(self.weight_hidden_to_output.T, self.error_output)

    def updateWeights(self):
        s_output = self.sigmoid(self.O_output) * (1 - self.sigmoid(self.O_output))
        s_hidden = self.sigmoid(self.O_hidden) * (1 - self.sigmoid(self.O_hidden))


        weight_correction_hidden_to_output = np.dot(self.learning_rate * self.error_output * s_output, self.O_hidden.T)
        weight_correction_input_to_hidden = np.dot(self.learning_rate * self.error_hidden * s_hidden, self.input.T)

        self.weight_input_to_hidden = self.weight_input_to_hidden + weight_correction_input_to_hidden
        self.weight_hidden_to_output = self.weight_hidden_to_output + weight_correction_hidden_to_output

    def sigmoid(self, input):
        return 1.0 / (1.0 + np.exp(-input))

    def train(self):
        with open('mnist_train (2).csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                self.input = np.asfarray([row[1:]])
                self.input = (((self.input / 255) * .990) + .010).T
                self.answer = self.createAnswer(row[0])
                self.answer_number = row[0]

                self.query()
                self.backPropagating()
                self.updateWeights()

                counter += 1
                if counter >= self.training_count:
                    break
            print("number of training: " + str(counter))

    def test(self):

        total = self.test_count
        correct = 0

        with open('mnist_test.csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                self.input = np.asfarray([row[1:]])
                self.input = (((self.input / 255) * .990) + .010).T
                self.answer = self.createAnswer(row[0])
                self.answer_number = row[0]

                self.query()

                print(self.answer_number)

                if int(self.answer_number) == int(self.O_output.argmax()):
                    correct += 1

                print(np.around(self.O_output.T, 3))
                print("-------------------------")

                counter += 1
                if counter >= self.test_count:
                    break

            print("algorithm percentage " + str(correct / total))

    @staticmethod
    def createAnswer(row):
        list = []
        i = 0
        while i < 10:
            if int(row) == i:
                list.append(.99)
            else:
                list.append(.01)
            i += 1

        array = np.array([list]).T
        return array


numberRecognition = NumberRecognition(784, 100, 10, .01)
numberRecognition.setTrainingTest(60000, 10000)
numberRecognition.train()
numberRecognition.test()

# Training Test alpha
# 60000,10000, 0.00234 => .88%
# 60000,10000, 0.003   => .8903%
# 60000,10000, 0.01    => .8945%

# 60000,100, 0.00234   => .93%
# 60000,10000, 0.003   => .93%
# 60000,100, 0.01      => .92%
# 10000,100, 0.01      => .91%

# 500,100, .07         => .80%
# 500,100, .07         => .81%
# 100,100, .3          => .66%

#/mnist_test.csv
#/mnist_train (2).csv

