require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')

const plot = require('node-remote-plot')
const _ = require('lodash')

const LogisticRegression = require('./logistic-regression')

const mnist = require('mnist-data')

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

const loadData = () => {
    
    const mnistData = mnist.training(0,60000)

    const features = mnistData.images.values.map(x => x.flat())

    const encodeLabels = mnistData.labels.values.map(label => {
            const arr = new Array(10).fill(0)
            arr[label] = 1
            return arr
        })

    return { features, labels: encodeLabels }
}

const { features, labels } = loadData()

const regression = new LogisticRegression(features, labels, {
    learningRate: 1,
    iterations: 80,
    batchSize: 500
})

regression.train()


const testMnistData = mnist.testing(0,1000)

const testFeatures = flattenImages(testMnistData.images.values)
const testLabels = encodeLabels(testMnistData.labels.values)

const accuracy = regression.test(testFeatures, testLabels)

console.log(accuracy)

plot({
    x: regression.costHistory.reverse()

})

console.log(regression.costHistory)