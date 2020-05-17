require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')

const plot = require('node-remote-plot')
const _ = require('lodash')

const loadCSV = require('../load-csv')

const LogisticRegression = require('./logistic-regression')

const { features, labels, testFeatures, testLabels } = loadCSV(
	"../data/cars.csv",
	{
		dataColumns: ["horsepower", "displacement", "weight"],
		labelColumns: ["mpg"],
		splitTest: 50,
		shuffle: true,
		converters: {
			mpg: (value) => {
				const mpg = parseFloat(value)
				if (mpg < 15) {
					return [1, 0, 0]
				} else if (mpg < 30) {
					return [0, 1, 0]
				} else {
					return [0, 0, 1]
				}
			}
		}
	}
)

const logisticRegression = new LogisticRegression(features, _.flattenDepth(labels, 1), {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 100,
    decisionBoundary: 0.6
})

logisticRegression.train()
console.log(logisticRegression.test(testFeatures, _.flattenDepth(testLabels, 1)))
// logisticRegression.predict([
//     [88, 97, 1.07]
// ]).print()

plot({
    x: logisticRegression.costHistory.reverse(),
    xLabel: "iterations",
    yLabel: "cost"
})