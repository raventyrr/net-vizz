import matplotlib.pyplot as plt
import h5py
import numpy as np
import mpld3


def showNetwork(struct, weights, labels):
    nbLayers  = len(struct)
    networkXY = []
    colors = ['b', 'r']
    for layer in range(nbLayers):
        layerXY = []
        if layer != 4:
            sumWeightsNeuron = np.sum(np.abs(weights['layer_' + str(layer)]['param_0']), axis=1)
            maxSumWeights = np.max(sumWeightsNeuron)
        for neuron in range(struct[layer]):
            neuronXY = (layer*10, neuron-(struct[layer]-1)/2.)
            if layer != 4 :
                inputScatters = plt.scatter(neuronXY[0],neuronXY[1], alpha=(sumWeightsNeuron[neuron]/maxSumWeights)**2)
            else :
                inputScatters = plt.scatter(neuronXY[0],neuronXY[1], alpha=1)
            layerXY.append(neuronXY)
        networkXY.append(layerXY)
        tooltip = mpld3.plugins.PointLabelTooltip(inputScatters, labels=labels)
        
        if layer != 0:
            print(weights['layer_' + str(layer-1)]['param_0'].value)
            maxWeights = np.amax(np.abs(weights['layer_' + str(layer-1)]['param_0']))
            for neuronLayer in range(struct[layer]):
                for neuronLayerP in range(struct[layer-1]):
                    print(layer, neuronLayer, neuronLayerP, maxWeights)
                    
                    plt.plot([networkXY[layer][neuronLayer][0],networkXY[layer-1][neuronLayerP][0]],
                             [networkXY[layer][neuronLayer][1],networkXY[layer-1][neuronLayerP][1]],
                             #alpha=1-np.exp(-((weights['layer_' + str(layer-1)]['param_0'][neuronLayerP][neuronLayer])/3)**2)
                             alpha = (weights['layer_' + str(layer-1)]['param_0'][neuronLayerP][neuronLayer] / maxWeights)**2,
                             c = colors[int(weights['layer_' + str(layer-1)]['param_0'][neuronLayerP][neuronLayer] > 0)])
    mpld3.show()
if __name__ == "__main__":
    struct  = [68,100,100,100,5]
    weights = h5py.File("weights.h5", "r")
    labels = ['point {0}'.format(i + 1) for i in range(68)]
    
    showNetwork(struct,  weights, labels)
