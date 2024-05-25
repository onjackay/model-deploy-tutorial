#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <iostream>
#include <fstream>

using namespace nvinfer1;
using namespace nvonnxparser;

class Logger : public ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

int main()
{
    // create the builder
    IBuilder *builder = createInferBuilder(logger);
    if (!builder)
    {
        std::cerr << "Error: failed to create the builder." << std::endl;
        return 1;
    }

    // create the network
    INetworkDefinition *network = builder->createNetworkV2(0);

    // import the ONNX model using the ONNX parser
    IParser *parser = createParser(*network, logger);
    if (!parser)
    {
        std::cerr << "Error: failed to create the parser." << std::endl;
        return 1;
    }
    parser->parseFromFile("mnist.onnx", static_cast<int>(ILogger::Severity::kWARNING));
    for (int i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cerr << parser->getError(i)->desc() << std::endl;
    }

    // build the engine
    IBuilderConfig *config = builder->createBuilderConfig();
    if (!config)
    {
        std::cerr << "Error: failed to create the builder configuration." << std::endl;
        return 1;
    }
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 20);
    config->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, 48 << 10);
    IHostMemory *serializedModel = builder->buildSerializedNetwork(*network, *config);

    // destroy the network and the parser
    delete parser;
    delete network;
    delete config;
    delete builder;

    // save the engine
    std::ofstream engineFile("mnist.engine", std::ios::binary);
    if (!engineFile)
    {
        std::cerr << "Error: failed to open the engine file." << std::endl;
        return 1;
    }
    engineFile.write(static_cast<char *>(serializedModel->data()), serializedModel->size());
    delete serializedModel;

    return 0;
}