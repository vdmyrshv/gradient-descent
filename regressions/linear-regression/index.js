require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const plot = require('node-remote-plot')

const LinearRegression = require('./linear-regression')

const loadCSV = require('../load-csv')

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    dataColumns:['horsepower', 'displacement', 'weight'],
    labelColumns:['mpg'],
    shuffle: true,
    splitTest: 50
})

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 10,
    batchSize: 10
})

regression.train()
const r2 = regression.test(testFeatures, testLabels)

regression.predict([
    [120, 380, 2]
]).print()

plot({
    x: regression.mseHistory.reverse(),
    xLabel: 'iteration number',
    yLabel: 'mse'
})

console.log(`updated M is: ${regression.weights.get(1,0)}`)
console.log(`updated B is: ${regression.weights.get(0,0)}`)

console.log(r2)

//console.log(`prediction: ${prediction}`)

