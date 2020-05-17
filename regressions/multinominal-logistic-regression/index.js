require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')

const plot = require('node-remote-plot')

const loadCSV = require('../load-csv')

const LogisticRegression = require('./logistic-regression')

const {features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['passedemissions'],
    splitTest: 50,
    shuffle: true,
    converters: {
        passedemissions: (value) => value==="TRUE"?1:0
    }
})

const logisticRegression = new LogisticRegression(features, labels, {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 50,
    decisionBoundary: 0.6
})

logisticRegression.train()
logisticRegression.predict([
    [88, 97, 1.07]
]).print()

plot({
    x: logisticRegression.costHistory.reverse(),
    xLabel: "iterations",
    yLabel: "cost"
})

console.log(logisticRegression.test(testFeatures, testLabels))