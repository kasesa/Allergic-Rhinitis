from pyFile import ARModel


modelAvl = ["vgg16","vgg19","inception","xception","resnet50","resnet101","densenet","inceptionResnet"]
modelSel = ["inception","xception"]
modelOnly = ["xception"]

exeState = "normal"

if exeState == "normal":
    model = ARModel()
    model.loadImages(r'C:\Users\cvpr\Documents\Bishal\Allergic Rhinitis\Dataset\rotate', plotType="rotated", 
        correctColor=False, contours=False, crop=True ,printImgDemo=False)
    model.prepareData()
    model.setDataAugmentation(rotation=20, zoom=0.15, wShift=0.2, hShift=0.2, shear=0.15, hFlip=True)
    model.setPartition(testSize=0.20)

    for currentModel in modelSel:
        model.setBaseModel(currentModel)
        model.setHeadModel(dropoutRate=0.5)
        model.initModel()
        model.setHyperParameters(learningRate = 1e-3, epochs = 200, batchSize = 8)
        model.compileModel()
        model.startTraining()
        model.startTesting()
        model.evalModel()
        model.generatePlot(iter=3)

if exeState == "crossvalidation":
    model = ARModel()
    for currentModel in modelSel:
        model.crossValidate(path=r'C:\Users\cvpr\Documents\Bishal\Allergic Rhinitis\Dataset', modelType=currentModel, 
                    dropoutRate=0.5, batchSize=8, epochs=100, learningRate=1e-3, iter=3, dataType = "all")