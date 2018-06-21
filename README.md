# MNN-with-TensorFlow

## Multilayer neural network using TensorFlow software developed by Google. 
The main goal of this assignment is to construct a multilayer neural network using TensorFlow software developed by Google. The target, was to analyse the program so it would be producing a result with a minimum of 0.01% error rate. 
The program adjusts its weights accordingly to the provided data and defined variables. The data was trained by having a set of 500 samples and dedicating 100 samples to the evaluation and test data.  The data was provided through Moodle page containing two different files, one of them being train.cvs and the other eval.cvs. These files were written in Excel to fit the data. 

## Main processing
When the program was firstly launched, several error occurrences were observed and fixed before proceeding further.  When all the errors were fixed, testing began using the training data. 

## Initial observation
The program contained several different parameters which were defining the data processing and evaluation. Then input and output files were defined which took the training data in and when it was processed though, the end was put into the output folder. 
Later, several arrays are seen which were filled with the variables from the datasets. Then different layers of placeholders were defined. After, two variables were initialised which were weights that would theoretically should adjust themselves accordingly to the inputs and variable entries. Then two Bias nodes were created, which were monitoring how flexible the perception is. Thus, providing a more accurately predicted data.
Then a hidden layer was defined with Sigmoid method, which took the weights and the bias that have been defined as parameters for adjusting the weights. 
Finally, initial cost was created to see how expensive the algorithm is and how efficient it can get depending on the variables adjusted. 
After all the base values initialised and enabled, session is created, which allowed the parameters to be persisted and adjusted within that session. The training data was trained the number of epochs that have been defined at the very start of the program as variables. The data was then printed which showed the hypothesised value, the target and the error occurrence between the values. Finally, when the for loop has finished, the answer has been created which was looking at the hypothesis for the Y - axis. By that accuracy was then formed by casting the answer as a float value. 
