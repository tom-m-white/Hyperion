#ifndef NN_INFERENCE_HPP
#define NN_INFERENCE_HPP

#include <string>
#include <vector>
#include <memory>
#include <torch/script.h>
#include <torch/torch.h>

// Forward-declare Position to keep headers clean
namespace hyperion {
namespace core {
    class Position;
}
}

// Result struct
struct InferenceResult {
    float value;
    std::vector<float> policy;
};

// Class Declaration
class NeuralNetwork {
public:
    explicit NeuralNetwork(const std::string& model_path);
    ~NeuralNetwork();

    InferenceResult infer(const hyperion::core::Position& pos);

private:
    torch::Tensor position_to_tensor(const hyperion::core::Position& pos);
    std::unique_ptr<torch::jit::Module> module_;
    torch::Device device_; 
};

#endif // NN_INFERENCE_HPP