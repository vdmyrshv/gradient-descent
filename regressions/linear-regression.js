//first require is not needed - that is only needed in the root or main file to tell the app that node is the framework being used
const tf = require("@tensorflow/tfjs")

const _ = require("lodash")

class LinearRegression {
	constructor(features, labels, options) {
		//instance variables
		this.features = this.processFeatures(features)
		this.labels = tf.tensor(labels)

		//the first value in the shape property is the number of rows, thans why the [0] is there
		//remember that any operation on a tensor will return a new tensor, tensors are immutable
		// this is why the whole operation of adding a column of 1s is assigned back to this.features

		//also remember that the ones can be concatenated onto the END of the features column, the m/b weights tensor will have
		//to have its order reversed
		//this.features = this.processFeatures(this.features)

		//this.options = Object.assign({learningRate: 0.1, }, options)
		this.options = { learningRate: 0.1, iterations: 1000, ...options }

		this.weights = tf.zeros([this.features.shape[1], 1])

		this.mseHistory = []
	}

	// //old version of gradient descent using arrays
	// gradientDescent(){
	//     const currentGuessesForMPG = this.features.map(row=> {
	//         return this.m*row[0] + this.b
	//     })

	//     const bSlope =
	// 		(_.sum(
	// 			currentGuessesForMPG.map(
	// 				(guess, i) => guess - this.labels[i][0]
	// 			)
	// 		) *
	// 			2) /
	//         this.features.length

	//     const mSlope =
	// 		(_.sum(
	// 			currentGuessesForMPG.map(
	// 				(guess, i) =>
	// 					-1 * this.features[i][0] * (this.labels[i][0] - guess)
	// 			)
	// 		) *
	// 			2) /
	//         this.features.length

	//     this.m = this.m - mSlope*this.options.learningRate
	//     this.b = this.b - bSlope*this.options.learningRate

	// }

	gradientDescent() {
		const currentGuesses = this.features.matMul(this.weights)
		const differences = currentGuesses.sub(this.labels)
		const slopes = this.features
			.transpose()
			.matMul(differences)
			.div(this.features.shape[0]) //remember, [0] is to get the rows, or the first value int he shape

		this.weights = this.weights.sub(slopes.mul(this.options.learningRate))
	}

	train() {
		for (let i = 0; i < this.options.iterations; i++) {
			this.gradientDescent()
			this.recordMSE()
			this.updateLearningRate()
		}
	}

	test(testFeatures, testLabels) {
		testFeatures = this.processFeatures(testFeatures)
		testLabels = tf.tensor(testLabels)

		const predictions = testFeatures.matMul(this.weights)

		const res = testLabels.sub(predictions).pow(2).sum().get() //dont have to pass any args to get because its just one number in the tensor
		const tot = testLabels.sub(testLabels.mean()).pow(2).sum().get() //.mean method gets the avg of a tensor

		return 1 - res / tot
	}

	processFeatures(features) {
		features = tf.tensor(features)

		if (this.mean && this.variance) {
			features = features.sub(this.mean).div(this.variance.pow(0.5))
		} else {
			features = this.standardize(features)
		}

		features = tf.ones([features.shape[0], 1]).concat(features, 1)

		return features
	}

	standardize(features) {
		const { mean, variance } = tf.moments(features, 0)

		//NOTE: you can create instance variables outside of the constructor as you see here
		//even though the instance variables are created here, they're "global" class variables
		this.mean = mean
		this.variance = variance

		return features.sub(mean).div(variance.pow(0.5))
	}

	recordMSE() {
		const mse = this.features
			.matMul(this.weights)
			.sub(this.labels)
			.pow(2)
			.sum()
			.div(this.features.shape[0])
			.get()

		//using unshift instead of push can make it easier to access the most recently added values to an array
		//for example see below in updateLearningRate(), instead of putting array.length-1 in property accessor, just use 0 and 1 indices
		this.mseHistory.unshift(mse)
	}

	updateLearningRate() {
		//console.log(this.options.learningRate)
		// if(this.mseHistory.length < 2){
		//     return
		// }
		//technically shouldn't even need the if statement above, since if the array index is undefined
		//in a conditional it evaluates to NaN and is automatically false
		if (this.mseHistory[0] > this.mseHistory[1]) {
			this.options.learningRate /= 2
		} else if (this.mseHistory[0] <= this.mseHistory[1]) {
			this.options.learningRate *= 1.05
		}
	}
}

module.exports = LinearRegression
