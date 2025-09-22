import asyncio

from .data import loadTasks
from .utils import getCachedFileJsonAsync, doesCachedFileJsonExistOrInProgress
from .router import getParams

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

modelsToStudy = OPENWEIGHT_MODELS + ANTHROPIC_MODELS + OPENAI_MODELS
modelsToStudy = [("Qwen/Qwen2.5-7B-Instruct", "vllm")] # testing


async def getTaskOutputs(nRolloutsPerPrompt, maxInferenceTokens=10000, batchSize=1000):
    def modelStrToOutputPath(modelStr):
        return f"taskOutputs/{modelStr}.json"
    
    def evalFunc(mdelId, inferenceType):
        tasks = loadTasks()
        router = getRouter(modelId, inferenceType, tensorizeModels=tensorizeModels)
        evalInfo = getEvalInfo(modelId, inferenceType, evalType="taskOutputs")
        tokenizeParams, inferenceParams = getParams(modelId, inferenceType, evalInfo, maxInferenceTokens):
        
        def getInputsFunc(task):
            return [Prompt(messages=[ChatMessage(content=task, role=MessageRole.user)]) for _ in range(nRolloutsPerPrompt)]

        async def processBatchFunc(inputBatch):
            return await router.processPrompts(inputBatch, tokenizeParams, **inferenceParams)

        def processOutputFunc(prompt, modelInputs, outputs):
            results = []
            # Add tool calls to output text
            for output in outputs:
                text = output[0].completion
                results.append(text)
            return results

        return await runBatchedAsync(inputs=list(range(len(prompts))),
                                getInputs=getInputsFunc,
                                processBatch=processBatchFunc,
                                processOutput=processOutputFunc,
                                batchSize=batchSize,
                                noCancel=True)

    outputs = await getOutputs(modelStrToOutputPath=modelStrToOutputPath, evalFunc=evalFunc)
    if outputs == "return": return
    

async def getOutputs(modelStrToOutputPath, evalFunc):
    for modelId, inferenceType in modelsToStudy:
        outputPath = modelIdToOutputPath(modelId.replace('/', '_'))

        if not doesCachedFileJsonExistOrInProgress(outputPath):
            modelEvalFunc = functools.partial(evalFunc, modelId=modelId, inferenceType=inferenceType)
            await getCachedFileJsonAsync(outputPath, modelEvalFunc)
            # run this over and over to get all of them, we need to bail so vllm properly cleans up
            return "return"
    
    results = {}
    for modelId, inferenceType in modelsToStudy:
        outputPath = modelIdToOutputPath(modelId)
        results[modelId] = getCachedFileJsonAsync(outputPath, lambda: None)
    return results


    

if __name__ == "__main__":
    asyncio.run(getTaskOutputs())