require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')

const plot = require('node-remote-plot')
const _ = require('lodash')

const LogisticRegression = require('./logistic-regression')

const mnist = require('mnist-data')

const mnistData = mnist.training(0,5000)

const flattenImages = (images) => {
    return images.map(x => x.flat())
}

const encodeLabels = (labels) => {
    return labels.map(label => {
        const arr = new Array(10).fill(0)
        arr[label] = 1
        return arr
    })
}

const features = flattenImages(mnistData.images.values)

const encodedLabels = encodeLabels(mnistData.labels.values)

const regression = new LogisticRegression(features, encodedLabels, {
    learningRate: 1,
    iterations: 20,
    batchSize: 100
})

regression.train()

const testMnistData = mnist.testing(0,100)

const testFeatures = flattenImages(testMnistData.images.values)
const testLabels = encodeLabels(testMnistData.labels.values)

const accuracy = regression.test(testFeatures, testLabels)

console.log(accuracy)