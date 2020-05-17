//first require is not needed - that is only needed in the root or main file to tell the app that node is the framework being used
const tf = require("@tensorflow/tfjs")

const _ = require("lodash")

class LogisticRegression {
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
		this.options = { learningRate: 0.1, iterations: 1000, decisionBoundary: 0.5, ...options }

		this.weights = tf.zeros([this.features.shape[1], 1])

		this.costHistory = []
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

	gradientDescent(features, labels) {
		//THE ONLY CHANGE YOU HAVE TO DO IS CHAIN ON .SIGMOID!!! .sigmoid method performs an elementwise operation on the tensor multiplying by sigmoid
		const currentGuesses = features.matMul(this.weights).sigmoid()
		const differences = currentGuesses.sub(labels)
		const slopes = features
			.transpose()
			.matMul(differences)
			.div(features.shape[0]) //remember, [0] is to get the rows, or the first value int he shape

		this.weights = this.weights.sub(slopes.mul(this.options.learningRate))
	}

	train() {
		const { batchSize } = this.options
		const batchQuantity = Math.floor(this.features.shape[0] / batchSize)
		for (let i = 0; i < this.options.iterations; i++) {
			for (let j = 0; j < batchQuantity; j++) {
				const featureSlice = this.features.slice(
					[batchSize * j, 0],
					[batchSize, -1]
				)
				const labelSlice = this.labels.slice(
					[batchSize * j, 0],
					[batchSize, -1]
				)
				this.gradientDescent(featureSlice, labelSlice)
			}
			this.recordCost()
			this.updateLearningRate()
		}
	}

	predict(observations) {

        const { decisionBoundary } = this.options

		//call sigmoid method here as well
		return this.processFeatures(observations)
			.matMul(this.weights)
			.sigmoid()
            .greater(decisionBoundary)
            .cast('float32') //tells tf to treat the output of the .greater() method as a float32 as opposed to a boolean
            //the .round() method would be here if the decision boundary was simply 50%, but .greater() allows for manual manipulation of the decision boundary, the passed argument inside is the prob at which 1
            //DEBUGGING: also, when using the .greater() method, tensorflow will regard the outputs as booleans
	}

	test(testFeatures, testLabels) {
		const predictions = this.predict(testFeatures)
		testLabels = tf.tensor(testLabels)
		const incorrect = predictions.sub(testLabels).abs().sum().get()

		return (predictions.shape[0] - incorrect) / predictions.shape[0]
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

	recordCost() {
        const guesses = this.features.matMul(this.weights).sigmoid()

        const termOne = this.labels.transpose().matMul(guesses.log())
        const termTwo = this.labels.mul(-1).add(1).transpose().matMul(
            guesses.mul(-1).add(1).log()
        )

        const cost = termOne.add(termTwo).div(this.features.shape[0]).mul(-1).get(0,0)

		//using unshift instead of push can make it easier to access the most recently added values to an array
		//for example see below in updateLearningRate(), instead of putting array.length-1 in property accessor, just use 0 and 1 indices
		this.costHistory.unshift(cost)
	}

	updateLearningRate() {
		console.log(this.options.learningRate)
		// if(this.mseHistory.length < 2){
		//     return
		// }
		//technically shouldn't even need the if statement above, since if the array index is undefined
		//in a conditional it evaluates to NaN and is automatically false
		if (this.costHistory[0] > this.costHistory[1]) {
			this.options.learningRate /= 2
		} else if (this.costHistory[0] <= this.costHistory[1]) {
			this.options.learningRate *= 1.05
		}
	}
}

module.exports = LogisticRegression
