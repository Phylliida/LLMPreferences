import asyncio
import re
import functools
import numpy as np

from .data.tasks import loadTasks
from .utils import getCachedFileJsonAsync, doesCachedFileJsonExistOrInProgress, runBatchedAsync
from .router import getParams, getRouter
from safetytooling.data_models import Prompt, ChatMessage, MessageRole, LLMResponse

# This is the stuff that changes for different kinds of eval
def getEvalInfo(modelName, inferenceType, evalType):
    # default values
    evalInfo = {
        "tools": None,
        "prefixMessages": [],
        "appendToSystemPrompt": None,
        "processData": None,
        "addBailPrompt": None,
        "evalType": evalType,
    }
    return evalInfo

OPENWEIGHT_MODELS = [

    # Llamas
    ("NousResearch/Hermes-3-Llama-3.2-3B", "vllm"),
    ("NousResearch/Hermes-3-Llama-3.1-8B", "vllm"),
    ("unsloth/Llama-3.1-8B-Instruct", "vllm"),

    # Qwen
    ("Qwen/Qwen3-1.7B", "vllm"),
    ("Qwen/Qwen3-4B", "vllm"),
    ("Qwen/Qwen3-8B", "vllm"),
    ("Qwen/Qwen3-32B", "vllm"),
    ("Qwen/Qwen3-30B-A3B", "vllm"),
    ("Qwen/QwQ-32B", "vllm"),
    ("Qwen/Qwen2.5-7B-Instruct", "vllm"),
    
    # GLM
    ("zai-org/GLM-4-32B-0414", "vllm"),
    
    # Gemma
    ("google/gemma-2-2b-it", "vllm"),
    ("google/gemma-2-9b-it", "vllm"),
    ("google/gemma-2-27b-it", "vllm"),
    
    # Deepseek
    #("deepseek/deepseek-r1", "openrouter"),
]


ANTHROPIC_MODELS = [
    ("claude-3-haiku-20240307", "anthropic"),
    ("claude-3-5-haiku-20241022", "anthropic"),

    ("claude-3-5-sonnet-20240620", "anthropic"),
    ("claude-3-5-sonnet-20241022", "anthropic"),
    ("claude-3-7-sonnet-20250219", "anthropic"),
    ("claude-sonnet-4-20250514", "anthropic"),
    
    ("claude-3-opus-20240229", "anthropic"),
    ("claude-opus-4-20250514", "anthropic"),
    ("claude-opus-4-1-20250805", "anthropic"),
]

OPENAI_MODELS = [
    ("gpt-3.5-turbo", "openai"),
    ("gpt-4", "openai"),
    ("gpt-4-turbo", "openai"),
    ("gpt-4o-mini", "openai"),
    ("gpt-4o", "openai"),
    ("gpt-4.1-nano", "openai"),
    ("gpt-4.1-mini", "openai"),
    ("gpt-4.1", "openai"),
    #('gpt-4.5-preview', "openai"),
    #("o4-mini", "openai"),
    #("o3-mini", "openai"),
    #("o3", "openai"),
    #("o1-mini", "openai"),
    #("o1-preview", "openai"),
    #("o1", "openai"),
]

modelsToStudy = ANTHROPIC_MODELS + OPENAI_MODELS + OPENWEIGHT_MODELS
modelsToStudy = [("Qwen/Qwen2.5-7B-Instruct", "vllm"), ("google/gemma-2-9b-it", "vllm")]#, ("zai-org/GLM-4-32B-0414", "vllm")] # testing


async def getTaskOutputs(nRolloutsPerPrompt, maxInferenceTokens=8000, batchSize=1000):
    def modelStrToOutputPath(modelStr):
        return f"taskOutputs/{modelStr}.json"
    
    async def evalFunc(modelId, inferenceType):
        tasksTrain, tasksTest = loadTasks()
        tasks = tasksTrain + tasksTest
        return await getRollouts(prompts=tasks, nRolloutsPerPrompt=nRolloutsPerPrompt, modelId=modelId, inferenceType=inferenceType, evalType="taskOutputs", maxInferenceTokens=maxInferenceTokens, batchSize=batchSize)
    outputs = await getOutputs(modelStrToOutputPath=modelStrToOutputPath, evalFunc=evalFunc)
    if outputs == "return": return "return"
    
async def getTaskTokenEstimates(nRolloutsPerPrompt, chainOfThought, evalName, train=True, maxInferenceTokens=8000, batchSize=1000):
    def modelStrToOutputPath(modelStr):
        return f"{evalName}/{modelStr}.json"
    
    async def evalFunc(modelId, inferenceType):
        tasksTrain, tasksTest = loadTasks()
        def taskToEstimate(task):
            return f"Please estimate how many output tokens this task will take to complete.\nTask: {task}\nHow many output tokens will you use to complete this task? Answer in this format:\n{chainOfThought}<tokenEstimate>\nEstimated number of output tokens needed to complete task\n</tokenEstimate>"
        tokenEstimates = list(map(taskToEstimate, tasksTrain if train else tastsTest))
        return await getRollouts(prompts=tokenEstimates, nRolloutsPerPrompt=nRolloutsPerPrompt, modelId=modelId, inferenceType=inferenceType, evalType=evalName, maxInferenceTokens=maxInferenceTokens, batchSize=batchSize)
    outputs = await getOutputs(modelStrToOutputPath=modelStrToOutputPath, evalFunc=evalFunc)
    if outputs == "return": return "return"

async def getRollouts(prompts, nRolloutsPerPrompt, modelId, inferenceType, evalType, maxInferenceTokens, batchSize):

    router = getRouter(modelId, inferenceType)
    evalInfo = getEvalInfo(modelId, inferenceType, evalType=evalType)
    tokenizeParams, inferenceParams = getParams(modelId, inferenceType, evalInfo, maxInferenceTokens)
    
    def getInputsFunc(prompt):
        return [Prompt(messages=[ChatMessage(content=prompt, role=MessageRole.user)]) for _ in range(nRolloutsPerPrompt)]

    async def processBatchFunc(inputBatch):
        return await router.processPrompts(inputBatch, tokenizeParams, **inferenceParams)

    def processOutputFunc(prompt, modelInputs, outputs):
        results = []
        # Add tool calls to output text
        for output in outputs:
            text = output[0].completion
            text_tokens = output[0].usage.output_tokens
            results.append({"text": text, "output_tokens": text_tokens})
        return results

    return await runBatchedAsync(inputs=prompts,
                            getInputs=getInputsFunc,
                            processBatch=processBatchFunc,
                            processOutput=processOutputFunc,
                            batchSize=batchSize,
                            noCancel=True)



async def getOutputs(modelStrToOutputPath, evalFunc):
    for modelId, inferenceType in modelsToStudy:
        outputPath = modelStrToOutputPath(modelId.replace('/', '_'))

        if not doesCachedFileJsonExistOrInProgress(outputPath):
            modelEvalFunc = functools.partial(evalFunc, modelId=modelId, inferenceType=inferenceType)
            await getCachedFileJsonAsync(outputPath, modelEvalFunc)
            # run this over and over to get all of them, we need to bail so vllm properly cleans up
            return "return"
    
    results = {}
    for modelId, inferenceType in modelsToStudy:
        outputPath = modelStrToOutputPath(modelId.replace('/', '_'))
        results[modelId] = await getCachedFileJsonAsync(outputPath, lambda: None)
    return results

def extractTokenEstimate(text):
    text = text.lower() # don't be case sensitive
    text = re.sub(r"\s+", "", text) # ignore spaces cause they don't matter and may mess up the tag
    estimateTag = "<tokenestimate>"
    posOfStartOfEstimate = text.find(estimateTag)
    if posOfStartOfEstimate == -1: # didn't give any estimate, bail
        return None
    estimateText = text[posOfStartOfEstimate + len(estimateTag):]
    endOfEstimate = max(estimateText.find("<t"), estimateText.find("<\\t"), estimateText.find("</t"), estimateText.find("<//t"))
    if endOfEstimate == -1:
        return None
    estimateText = estimateText[:endOfEstimate]
    numbers = list(map(float, re.findall(r"\d+\.?\d*", text)))
    return np.mean(np.array(numbers)) # if multiple, return mean (like if it gave a range 100-200)
    

if __name__ == "__main__":
    resOutputs = asyncio.run(getTaskOutputs(nRolloutsPerPrompt=10))
    if resOutputs != "return":
        resEstimates = asyncio.run(getTaskTokenEstimates(chainOfThought="", nRolloutsPerPrompt=10, evalName="estimates"))
        if resEstimates != "return":
            resEstimatesCot = asyncio.run(getTaskTokenEstimates(chainOfThought="<reasoning>\nChain of thought used to determine estimate\n</reasoning>\n", nRolloutsPerPrompt=10, evalName="estimatesCOT"))
            if resEstimatesCot != "return":
                for model, outputs in resOutputs.items():
                    estimates = resEstimates[model]
                    estimatesCot = resEstimates[model]
                    tasksTrain, tasksTest = loadTasks()
                    diffs = []
                    diffsCot = []
                    for prompt, promptOutputs, estimateOutputs, estimateCotOutputs in enumerate(zip(tasksTrain, outputs, estimates, estimatesCot)):
                        averageTokensUsed = np.mean(np.array([output['output_tokens'] for output in promptOutputs]))
                        averageEstimates = [est for est in [extractTokenEstimate(output['text']) for output in estimateOutputs] if not est is None]
                        averageCotEstimates = [est for est in [extractTokenEstimate(output['text']) for output in estimateCotOutputs] if not est is None]
                        averageEstimate = np.mean(np.array(averageEstimates)) if len(averageEstimates) > 0 else None
                        averageCotEstimate = np.mean(np.array(averageCotEstimates)) if len(averageCotEstimates) > 0 else None
                        diffs.append((averageEstimate - averageTokensUsed) if not averageEstimate is None else None)
                        diffsCot.append((averageCotEstimate - averageTokensUsed) if not averageCotEstimate is None else None)
                    print(model)
                    print("diffs")
                    print(diffs)
                    print("diffs cot")
                    print(diffsCot)
                    print("diffs mean")
                    print(np.mean([x for x in diffs if not x is None]))
                    print("diffs cot mean")
                    print(np.mean([x for x in diffsCot if not x is None]))