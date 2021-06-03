#this program demonstrates linear regression using gradient descent

import matplotlib.pyplot as plt

#define a function that implements our "model"
#in this case it's a simple linear model
def model(b0, b1, x):
	#compute the prediction based on the arguments
	#notice that the variables here are LOCAL to the "model" function
	y = b0 + b1*x
	return y

#test (use the function)
#print model(0, 2, 2.2)

#define our data points as lists
x = [1,2,3,4,5,6]
#target (observed values):
y = [1,3,3,2,5,7]

#set up our initial coeffecients (really a guess)
b0 = 0.0
b1 = 0.0
#learning rate is a "meta parameter" which affects how quickly or well
#the alogorithm "learns" - again, the initial value is a guess
learning_rate=0.01
max_epochs = 5

current_epoch = 0

while(current_epoch < max_epochs):
	#let's write a loop to take us through one epoch of learning
	#for each epoch, go through all the data points once
	# (for now, ignore shuffling the data)
	for i in range(0, len(x)):
		#compute the model error for this data point
		error = model(b0, b1, x[i]) - y[i]
		#adjust coefficients
		b0 = b0 - learning_rate*error
		b1 = b1 - learning_rate*error*x[i]
		print ("after point ", i, " b0 is now ", b0, " b1 is now ", b1)
	current_epoch += 1
	print ("finished epoch ", current_epoch)

#at this point learning is DONE. We're now evaluating the model (visually)
#at this point b0 and b1 are the "best" learned coefficients
#at this point we have completed ALL EPOCHS (for now, let's visualize the model)
#to visualize the predictions of the model, we will use the model
#to make predictions for each value of x
predictions = []
for i in range(0, len(x)):
	prediction = model(b0,b1,x[i])
	predictions.append(prediction)
	#print predictions
#try plotting the data
plt.scatter(x, y, color = 'black')
#add our predictions to the plot
plt.plot(x, predictions, color='blue', linewidth=2)
plt.show()
#TODO: implement the RMSE threshold termination condition
#TODO: shuffle the data each epoch BEFORE adjusting coefficients
#TODO: (for fun) try plotting the predicted values each epoch instead of at the end

