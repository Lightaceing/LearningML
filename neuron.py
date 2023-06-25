import math


"""
takes in neuron count , weights as list and target value || bias weight is set to 1 so it acts as adding 1
then need to invoke take_input fn. to provide input data and bias
then invoke optimize()
"""


class Adaline:

    weights = []
    inputs = []
    # sum value
    y = 0
    z = 0
    target = 0
    learning_rate = 0.2
    weight_change = 1

   # Init n neurons with weights 0 and last one as bias

    def __init__(self, input_count: int, weights: list, target: int):

        assert input_count > 0, "There cant be 0 neurons!"
        assert input_count == len(weights), "Not Enough weights!!"
        self.target = target
        self.n = input_count
        for i in range(self.n):
            self.weights.append(weights[i])
        self.weights.append(1)

    # taking in input data as a list
    def take_input(self, input_data: list, bias: int):

        assert (len(self.weights) == len(
            input_data)+1), f"Not enough input data members!!"

        for i in range(self.n):
            self.inputs.append(input_data[i])

        # bias
        self.inputs.append(bias)

    # computing y = sum(W*X+b)
    def computeY(self):
        sum = 0
        for i in range(self.n+1):
            sum += self.weights[i]*self.inputs[i]
        self.y = sum

    # applying activation fn. 'Sigmoid'
    def apply_activation(self):
        self.z = 1/(1 + math.exp(-1*self.y))

    # weights are updated
    def update_weights(self):
        error = (self.target - self.z)
        change = self.learning_rate*(error)
        for i in range(len(self.weights)-1):
            self.weights[i] = self.weights[i] + change
        return change

    # o/p is calculated and weights in computed till change in weights is less than minimum_change or it reaches max_epoch
    def optimize(self, max_epoch, minimum_change):
        for i in range(max_epoch):
            if (self.weight_change < minimum_change):
                break
            else:
                self.computeY()
                self.apply_activation()
                self.weight_change = self.update_weights()
            print(
                f"Count : {i+1} Weights : {self.weights}, Weight change :  {self.weight_change}")
        print(
            f"Final output weight is : {self.weights}, at epoch no : {i+1} \nError is : {self.weight_change/self.learning_rate}")


ad = Adaline(7, [0, 0, 0, 0, 0, 0, 0], 1)
ad.take_input([1, 1, 0, 2, 1, 3, -1], 1)
print(ad.optimize(100, 0.01))
