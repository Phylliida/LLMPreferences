
import trueskill
import random
import matplotlib.pyplot as plt
import numpy as np
import functools
import torch
import math
from collections import defaultdict

def generateComparisonsGraph():
    global numCompares
    global compareCache
    random.seed(27)
    
    algorithms = { }
    
    for i in np.linspace(10, 300, 20):
        algorithms[f"trueskill {i}"] = (int(i), functools.partial(batchedTrueskillBetter))    
    numComparisons = []
    fig, ax = plt.subplots()
    for algorithmName, (numData, algorithm) in algorithms.items():
        print(algorithmName)
        scoresFromAllRuns = []
        numComparisonsFromAllRuns = []
        for runs in range(5):
            print(runs)
            compareCache = {}
            numCompares = 0
            data = list(range(numData))
            random.shuffle(data)
            correctRanking = torch.argsort(torch.tensor(data))
            ranking = torch.arange(0, numData, dtype=torch.long)
            def scoreRanking():
                return torch.mean(torch.abs(ranking - correctRanking).float()).item()
            scores = []    
            def lessThanFunc(a,b):
                global compareCache
                global numCompares
                result = 1.0 if a < b else 0.0
                numCompares += 1
                scores.append(scoreRanking())
                if random.random() < 0.1:
                    return 1.0-result
                else:
                    return result
            def lessThanFuncBatched(asAndBs):
                print(f"doing {len(asAndBs)} comparisons")
                return [lessThanFunc(a,b) for (a,b) in asAndBs]
            print(f"running for {len(data)} datapoints")
            algorithm(data=data, ranking=ranking, lessThanFuncBatched=lessThanFuncBatched)
            print([data[i] for i in ranking])
            scoresFromAllRuns.append(scores)
            numComparisonsFromAllRuns.append(numCompares)
            print(numCompares)
        numComparisons.append((numData, numComparisonsFromAllRuns))
        maxLenScores = len(max(scoresFromAllRuns, key=lambda x: len(x)))
        confidence = 0.95 # 0.99
        zValueForConfidence = 1.96 # 2.58
        minConfidence = []
        maxConfidence = []
        means = []
        for i in range(maxLenScores):
            scoresForComparisonI = torch.tensor([scores[i] for scores in scoresFromAllRuns if len(scores) > i])
            meanForComparisonI = torch.mean(scoresForComparisonI)
            stdForComparisonI = torch.std(scoresForComparisonI)
            # confidence interval = mean - zValue * std/sqrt(n)
            offset = zValueForConfidence * stdForComparisonI / math.sqrt(scoresForComparisonI.size()[0])
            means.append(meanForComparisonI)
            minConfidence.append(meanForComparisonI - offset)
            maxConfidence.append(meanForComparisonI + offset)
        x = np.arange(0, len(means))
        y = np.array(means)
        yMin = np.array(minConfidence)
        yMax = np.array(maxConfidence)
    
    
    zValueForConfidence = 1.96 # 0.95
    x = np.array([numData for (numData, numComps) in numComparisons])
    y = np.array([np.mean(numComps) for (numData, numComps) in numComparisons])
    lower = np.array([np.mean(numComps)-zValueForConfidence*np.std(numComps)/math.sqrt(len(numComps)) for (numData, numComps) in numComparisons])
    upper = np.array([np.mean(numComps)+zValueForConfidence*np.std(numComps)/math.sqrt(len(numComps)) for (numData, numComps) in numComparisons])
    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.fill_between(x, lower, upper, alpha=.3)
    ax.set_title("Number of comparisons until convergence for trueskill bracket size 2")
    ax.set_ylabel("Number of comparisons")
    ax.set_xlabel("Number of data points")
    plt.show()


def batchedTrueskillBetter(data, ranking, lessThanFuncBatched):
    global numComparisons

    elos = [trueskill.Rating(25) for _ in data]
    eloMeans = torch.tensor([elo.mu for elo in elos])
    
    doneSoFar = defaultdict(lambda: 0)
    randomInitialPairs = []
    # random initial pairing to estimate elos
    for randomIters in range(5):
        randomInitials = list(range(len(data)))
        random.shuffle(randomInitials)
        randomInitialPairs += [(randomInitials[i], randomInitials[len(data)-i-1]) for i in range(len(data)//2)]
    randomInitialPairsData = [(data[i], data[j]) for (i,j) in randomInitialPairs]
    for iIsLessThanJPr, (i,j) in zip(lessThanFuncBatched(randomInitialPairsData), randomInitialPairs):
        if iIsLessThanJPr < 0.5: # i wins
            elos[i], elos[j] = trueskill.rate_1vs1(elos[i], elos[j])
        elif iIsLessThanJPr > 0.5: # j wins
            elos[j], elos[i] = trueskill.rate_1vs1(elos[j], elos[i])
        eloMeans[i], eloMeans[j] = elos[i].mu, elos[j].mu
        doneSoFar[i,j] += 1
        doneSoFar[j,i] += 1

    numComparisons = 0
    offset = 0
    numTimesWithNoChanges = 0
        
    
    oldSortedIndices = torch.argsort(eloMeans)
    while True:
        pairs = []
        chosen = set()
        offset = (offset + 1) % 2
        sortedIndices = torch.argsort(eloMeans)
        if torch.all(oldSortedIndices == sortedIndices):
            numTimesWithNoChanges += 1
        else:
            numTimesWithNoChanges = 0
        oldSortedIndices = sortedIndices
        currentCompareBatch = []
        for i in range(len(data)//2-1,-1,-1):
            curI = i*2+offset # we toggle ab ab ab b then #a ba ba ba
            curJ = i*2+offset+1
            if curJ < len(data):
                currentI = sortedIndices[curI].item()
                currentJ = sortedIndices[curJ].item()
                if not (currentI, currentJ) in doneSoFar or (doneSoFar[currentI, currentJ] < 5):
                    currentCompareBatch.append((currentI, currentJ))
        currentCompareBatchData = [(data[i], data[j]) for (i,j) in currentCompareBatch]
        for iIsLessThanJPr, (currentI, currentJ) in zip(lessThanFuncBatched(currentCompareBatchData), currentCompareBatch):
            if iIsLessThanJPr < 0.5: # i wins
                elos[currentI], elos[currentJ] = trueskill.rate_1vs1(elos[currentI], elos[currentJ])
            elif iIsLessThanJPr > 0.5: # j wins
                elos[currentJ], elos[currentI] = trueskill.rate_1vs1(elos[currentJ], elos[currentI])
            eloMeans[currentI], eloMeans[currentJ] = elos[currentI].mu, elos[currentJ].mu
            doneSoFar[currentI, currentJ] += 1
            doneSoFar[currentJ, currentI] += 1
        
        ranking[:] = torch.argsort(eloMeans)
        numComparisons += 1

        if numComparisons > len(data)*len(data)*2: # we've brute forced it, stap
            return torch.argsort(eloMeans)
        if numTimesWithNoChanges > 5: # bail if no changes
            return torch.argsort(eloMeans)

def simpleTrueskillBetter(data, ranking, lessThanFunc):
    elos = [trueskill.Rating(25) for _ in data]
    eloMeans = torch.tensor([elo.mu for elo in elos])
    
    # random initial pairing to estimate elos
    randomInitials = list(range(len(data)))
    random.shuffle(randomInitials)
    randomInitialPairs = [(randomInitials[i], randomInitials[len(data)-i-1]) for i in range(len(data)//2)]
    for i,j in randomInitialPairs:
        iIsLessThanJPr = lessThanFunc(data[i], data[j])
        if iIsLessThanJPr < 0.5: # i wins
            elos[i], elos[j] = trueskill.rate_1vs1(elos[i], elos[j])
        elif iIsLessThanJPr > 0.5: # j wins
            elos[j], elos[i] = trueskill.rate_1vs1(elos[j], elos[i])
        eloMeans[i], eloMeans[j] = elos[i].mu, elos[j].mu
   
    numComparisons = 0
    offset = 0
    numTimesWithNoChanges = 0
        
    
    doneSoFar = set()
    oldSortedIndices = torch.argsort(eloMeans)
    while True:
        pairs = []
        chosen = set()
        offset = (offset + 1) % 2
        sortedIndices = torch.argsort(eloMeans)
        if torch.all(oldSortedIndices == sortedIndices):
            numTimesWithNoChanges += 1
        else:
            numTimesWithNoChanges = 0
        oldSortedIndices = sortedIndices
        for i in range(len(data)//2-1,-1,-1):
            curI = i*2+offset
            curJ = i*2+offset+1
            if curJ < len(data):
                currentI = sortedIndices[curI]
                currentJ = sortedIndices[curJ]
                if not (currentI, currentJ) in doneSoFar and not (currentJ, currentI) in doneSoFar:
                    #print(f"Comparing {curI} {curJ}")
                    iIsLessThanJPr = lessThanFunc(data[currentI], data[currentJ])
                    if iIsLessThanJPr < 0.5: # i wins
                        elos[currentI], elos[currentJ] = trueskill.rate_1vs1(elos[currentI], elos[currentJ])
                    elif iIsLessThanJPr > 0.5: # j wins
                        elos[currentJ], elos[currentI] = trueskill.rate_1vs1(elos[currentJ], elos[currentI])
                    eloMeans[currentI], eloMeans[currentJ] = elos[currentI].mu, elos[currentJ].mu
                    ranking[:] = torch.argsort(eloMeans)
                    numComparisons += 1
                    doneSoFar.add((currentI, currentJ))
            if numComparisons > len(data)*len(data)*2:
                return torch.argsort(eloMeans)
        if numTimesWithNoChanges > 5: # bail if no changes
            return torch.argsort(eloMeans)



async def getRollouts(nRolloutsPerPrompt, batchSize, modelId, inferenceType, evalInfo, maxInferenceTokens=1000):
    router = getRouter(modelId, inferenceType)
    tokenizeParams, inferenceParams = getParams(modelId, inferenceType, evalInfo, maxInferenceTokens)
    prompts = [x['content'] for x in getDataset(evalInfo)]

    def replaceEmpty(s):
        return s if s != "" else "<Refusal Classifier Activated>"
    def getInputsFunc(promptI):
        # bail prompt adds prefix from previous rollout
        if evalInfo['addBailPrompt'] is not None:
            return [Prompt(messages=[
                ChatMessage(content=prompts[promptI], role=MessageRole.user),
                ChatMessage(content=replaceEmpty(evalInfo['rollout'][promptI][rolloutJ]), role=MessageRole.assistant),
                ChatMessage(content=evalInfo['addBailPrompt'], role=MessageRole.user),
            ]) for rolloutJ in range(nRolloutsPerPrompt)]
        # otherwise, just simple stuff
        else:
            return [Prompt(messages=[ChatMessage(content=prompts[promptI], role=MessageRole.user)]) for _ in range(nRolloutsPerPrompt)]

    async def processBatchFunc(inputBatch):
        return await router.processPrompts(inputBatch, tokenizeParams, **inferenceParams)

    def processOutputFunc(prompt, modelInputs, outputs):
        results = []
        # Add tool calls to output text
        for output in outputs:
            text = output[0].completion
            for message in (output[0].generated_content if output[0].generated_content else []):
                if message.role == MessageRole.tool and message.content['tool_name'] == getBailToolName(evalInfo['evalType']):
                    text += '{"name": "' + getBailToolName(evalInfo['evalType']) + '", "arguments": {}}'
            results.append(text)
        return results

    modelOutputs = await runBatchedAsync(inputs=list(range(len(prompts))),
                              getInputs=getInputsFunc,
                              processBatch=processBatchFunc,
                              processOutput=processOutputFunc,
                              batchSize=batchSize,
                              noCancel=True)
    return modelOutputs
