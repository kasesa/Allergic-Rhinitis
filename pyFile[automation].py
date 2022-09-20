from pyFile import ARModel


modelAvl = ["vgg16","vgg19","inception","xception","resnet50","resnet101","densenet","inceptionResnet"]
modelSel = ["xception", "resnet101"]
modelOnly = ["xception"]
losses = ["bce", "cce", "focal", "scce", "kld"]
lossOnly = ["bce"]

exeState = "normal"

if exeState == "normal":
    model = ARModel()
    model.loadImages(r'C:\Users\cvpr\Documents\Bishal\Allergic Rhinitis\Dataset\non_rotate', plotType="NR", classification="multiclass", colorMode="RGB",
        correctColor=False, contours=False, crop=True ,printImgDemo=False)
    model.prepareData()
    model.setDataAugmentation(normalizeData=False, rotation=10, zoom=0.15, wShift=0.2, hShift=0.2, shear=0.15, hFlip=True, vFlip=True)
    model.setPartition(testSize=0.20)
    for currentModel in modelOnly:
        for loss in lossOnly:
            model.setBaseModel(currentModel)
            model.setHeadModel(dropoutRate=0.5)
            model.initModel()
            model.setHyperParameters(learningRate = 1e-3, epochs = 200, batchSize = 8)
            model.compileModel(loss=loss)
            model.startTraining()
            model.startTesting()
            model.evalModel()
            model.generatePlot(iterInfo=5)
        # model.getGradCams(type="test")

if exeState == "crossvalidation":
    model = ARModel()
    for currentModel in modelSel:
        model.crossValidate(path=r'C:\Users\cvpr\Documents\Bishal\Allergic Rhinitis\Dataset', modelType=currentModel, 
                    dropoutRate=0.5, batchSize=8, epochs=100, learningRate=1e-3, iter=3, dataType = "all")