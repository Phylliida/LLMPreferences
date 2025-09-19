import asyncio

from .utils import getCachedFileJsonAsync, doesCachedFileJsonExistOrInProgress, 
from trueskill import Rating, quality_1vs1, rate_1vs1




async def getEloScoresAllModels():
    for modelId, inferenceType in modelsToStudy:
        outputPath = getEloOutputPath(modelId)

        if not doesCachedFileJsonExistOrInProgress(outputPath):
            async def getEloScoresFunc():
                return await getEloScores(modelId=modelId,
                                          inferenceType=inferenceType)
            await getCachedFileJsonAsync(outputPath, getEloScoresFunc)
            # run this over and over to get all of them, we need to bail so vllm properly cleans up
            return

async def getEloScores(modelId, inferenceType):
    

def getEloOutputPath(modelId):
    return f"eloEval/{modelId.replace('/', '_')}.json"

modelsToStudy = [("Qwen2.5-7B-Instruct", "vllm")]

if __name__ == "__main__":
    asyncio.run(getEloScoresAllModels())