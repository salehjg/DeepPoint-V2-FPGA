#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "CWeightLoader.h"
CWeightLoader::CWeightLoader(CXilinxInfo *xilInfo) {
  m_ptrXilInfo = xilInfo;
  m_uWeightCount = 0;
  m_bIsLoaded = false;
}
void CWeightLoader::LoadWeightsFromDisk(std::string &weightsBaseDir,
                                        std::string &pathToTxtFnameList) {
  std::string line;
  int idx=-1;

  std::ifstream inFile(pathToTxtFnameList);
  m_uWeightCount = std::count(std::istreambuf_iterator<char>(inFile), std::istreambuf_iterator<char>(), '\n');
  m_ptrWeightsCpu  = new CTensor<float>*[m_uWeightCount];
  m_ptrWeightsXil  = new CTensorXil<float>*[m_uWeightCount];

  std::ifstream txtFile (pathToTxtFnameList);
  if (!txtFile.is_open()) {
    SPDLOG_LOGGER_ERROR(logger,"Failed to open text file (WeightsLoader::LoadFromDisk)");
    return;
  }

  while (std::getline(txtFile, line)) {
    idx++;
    std::string weight_npy_path = weightsBaseDir + line;
    m_mWeightNameToIndex.insert(std::make_pair(line, idx) );
    m_vNumpyBuff.push_back(cnpy::npy_load(weight_npy_path));
    std::vector<unsigned> __shape(m_vNumpyBuff.back().shape.begin(), m_vNumpyBuff.back().shape.end());

    m_ptrWeightsCpu[idx] = new CTensor<float>(
        __shape,
        m_vNumpyBuff.back().data<float>()
    );

    if(__shape.size()==1 && __shape[0]==0) continue;
    int bank = ResolveMemoryBank(PLATFORMS::XIL, line);
    m_ptrWeightsXil[idx] = new CTensorXil<float>(m_ptrXilInfo, __shape, m_vNumpyBuff.back().data<float>(), bank);
    auto tag = _ResolveTensorTagOclXilinx(line);
    m_ptrWeightsXil[idx]->SetTensorTag(tag);
  }
  m_bIsLoaded = true;
  txtFile.close();
}
CTensorBase *CWeightLoader::AccessWeights(PLATFORMS platform, std::string &name) {
  if(platform == PLATFORMS::CPU)
    return m_ptrWeightsCpu[m_mWeightNameToIndex[name]];
  else if (platform == PLATFORMS::XIL)
    return m_ptrWeightsXil[m_mWeightNameToIndex[name]];
  else
    assert(false);
}
int CWeightLoader::ResolveMemoryBank(PLATFORMS platform, std::string &name) {
  if(platform == PLATFORMS::XIL)
    return _ResolveMemoryBankOclXilinx(name);
  else
    assert(false);
}
int CWeightLoader::_ResolveMemoryBankOclXilinx(std::string &name) {
  assert(
    ConfigTaskConv2::BankIndex_inputTn == ConfigTaskConv2::BankIndex_weightTn &&
    ConfigTaskConv2::BankIndex_weightTn == ConfigTaskConv2::BankIndex_biasTn &&
    ConfigTaskConv2::BankIndex_biasTn == ConfigTaskConv2::BankIndex_outputTn
  );

  assert(
    ConfigTaskMatOps::BankIndex_inputTn1 == ConfigTaskMatOps::BankIndex_inputTn2 &&
    ConfigTaskMatOps::BankIndex_inputTn2 == ConfigTaskMatOps::BankIndex_outputTn
  );

  assert(
    ConfigTaskMatMul::BankIndex_inputTn1 == ConfigTaskMatMul::BankIndex_inputTn2 &&
    ConfigTaskMatMul::BankIndex_inputTn2 == ConfigTaskMatMul::BankIndex_outputTn
  );

  int indexConv2 = ConfigTaskConv2::BankIndex_inputTn;
  int indexMatOps = ConfigTaskMatOps::BankIndex_inputTn1;
  int indexMatMul = ConfigTaskMatMul::BankIndex_inputTn1;

  bool isConv[] = {
    name == "transform_net1.tconv1.weights.npy",
    name == "transform_net1.tconv1.biases.npy",
    name == "transform_net1.tconv2.weights.npy",
    name == "transform_net1.tconv2.biases.npy",
    name == "transform_net1.tconv3.weights.npy",
    name == "transform_net1.tconv3.biases.npy",
    name == "dgcnn1.weights.npy",
    name == "dgcnn1.biases.npy",
    name == "dgcnn2.weights.npy",
    name == "dgcnn2.biases.npy",
    name == "dgcnn3.weights.npy",
    name == "dgcnn3.biases.npy",
    name == "dgcnn4.weights.npy",
    name == "dgcnn4.biases.npy",
    name == "agg.weights.npy",
    name == "agg.biases.npy"
  };

  bool isMatOps[] = {
    name == "transform_net1.tconv1.bn.gamma.npy",
    name == "transform_net1.tconv1.bn.beta.npy",
    name == "transform_net1.tconv1.bn.transform_net1.tconv1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "transform_net1.tconv1.bn.transform_net1.tconv1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "transform_net1.tconv2.bn.gamma.npy",
    name == "transform_net1.tconv2.bn.beta.npy",
    name == "transform_net1.tconv2.bn.transform_net1.tconv2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "transform_net1.tconv2.bn.transform_net1.tconv2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "transform_net1.tconv3.bn.gamma.npy",
    name == "transform_net1.tconv3.bn.beta.npy",
    name == "transform_net1.tconv3.bn.transform_net1.tconv3.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "transform_net1.tconv3.bn.transform_net1.tconv3.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "transform_net1.tfc1.bn.gamma.npy",
    name == "transform_net1.tfc1.bn.beta.npy",
    name == "transform_net1.tfc1.bn.transform_net1.tfc1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "transform_net1.tfc1.bn.transform_net1.tfc1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "transform_net1.tfc2.bn.gamma.npy",
    name == "transform_net1.tfc2.bn.beta.npy",
    name == "transform_net1.tfc2.bn.transform_net1.tfc2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "transform_net1.tfc2.bn.transform_net1.tfc2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "dgcnn1.bn.gamma.npy",
    name == "dgcnn1.bn.beta.npy",
    name == "dgcnn1.bn.dgcnn1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "dgcnn1.bn.dgcnn1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "dgcnn2.bn.gamma.npy",
    name == "dgcnn2.bn.beta.npy",
    name == "dgcnn2.bn.dgcnn2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "dgcnn2.bn.dgcnn2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "dgcnn3.bn.gamma.npy",
    name == "dgcnn3.bn.beta.npy",
    name == "dgcnn3.bn.dgcnn3.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "dgcnn3.bn.dgcnn3.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "dgcnn4.bn.gamma.npy",
    name == "dgcnn4.bn.beta.npy",
    name == "dgcnn4.bn.dgcnn4.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "dgcnn4.bn.dgcnn4.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "agg.bn.gamma.npy",
    name == "agg.bn.beta.npy",
    name == "agg.bn.agg.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "agg.bn.agg.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "fc1.bn.gamma.npy",
    name == "fc1.bn.beta.npy",
    name == "fc1.bn.fc1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "fc1.bn.fc1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "fc2.bn.gamma.npy",
    name == "fc2.bn.beta.npy",
    name == "fc2.bn.fc2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "fc2.bn.fc2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "transform_net1.tfc1.biases.npy",
    name == "transform_net1.tfc2.biases.npy",
    name == "fc1.biases.npy",
    name == "fc2.biases.npy",
    name == "fc3.biases.npy"
  };

  bool isMatMul[] = {
    name == "transform_net1.tfc1.weights.npy",
    name == "transform_net1.tfc2.weights.npy",
    name == "fc1.weights.npy",
    name == "fc2.weights.npy",
    name == "fc3.weights.npy"
  };

  bool rsltConv = false;
  bool rsltMatOps = false;
  bool rsltMatMul = false;

  for(bool item:isConv){
    rsltConv = rsltConv | item;
  }

  for(bool item:isMatOps){
    rsltMatOps = rsltMatOps | item;
  }

  for(bool item:isMatMul){
    rsltMatMul = rsltMatMul | item;
  }

  if(rsltConv){
    SPDLOG_LOGGER_DEBUG(logger,"The weight tensor \"{}\" is considered to be related to the layer \"Conv2\" and will be transfered to the DDR bank {}", name, indexConv2);
    return indexConv2;
  }

  if(rsltMatOps){
    SPDLOG_LOGGER_DEBUG(logger,"The weight tensor \"{}\" is considered to be related to the layer \"MatOps\" and will be transferred to the DDR bank {}", name, indexMatOps);
    return indexMatOps;
  }

  if(rsltMatMul){
    SPDLOG_LOGGER_DEBUG(logger,"The weight tensor \"{}\" is considered to be related to the layer \"MatMul\" and will be transferred to the DDR bank {}", name, indexMatMul);
    return indexMatMul;
  }

  return -1; //the default bank
}
std::string CWeightLoader::_ResolveTensorTagOclXilinx(std::string &name) {
  assert(
    ConfigTaskConv2::BankIndex_inputTn == ConfigTaskConv2::BankIndex_weightTn &&
    ConfigTaskConv2::BankIndex_weightTn == ConfigTaskConv2::BankIndex_biasTn &&
    ConfigTaskConv2::BankIndex_biasTn == ConfigTaskConv2::BankIndex_outputTn
  );

  assert(
    ConfigTaskMatOps::BankIndex_inputTn1 == ConfigTaskMatOps::BankIndex_inputTn2 &&
    ConfigTaskMatOps::BankIndex_inputTn2 == ConfigTaskMatOps::BankIndex_outputTn
  );

  assert(
    ConfigTaskMatMul::BankIndex_inputTn1 == ConfigTaskMatMul::BankIndex_inputTn2 &&
    ConfigTaskMatMul::BankIndex_inputTn2 == ConfigTaskMatMul::BankIndex_outputTn
  );

  bool isConvWeight[] = {
    name == "transform_net1.tconv1.weights.npy",
    name == "transform_net1.tconv2.weights.npy",
    name == "transform_net1.tconv3.weights.npy",
    name == "dgcnn1.weights.npy",
    name == "dgcnn2.weights.npy",
    name == "dgcnn3.weights.npy",
    name == "dgcnn4.weights.npy",
    name == "agg.weights.npy"
  };

  bool isConvBias[] = {
    name == "transform_net1.tconv1.weights.npy",
    name == "transform_net1.tconv1.biases.npy",
    name == "transform_net1.tconv2.biases.npy",
    name == "transform_net1.tconv3.biases.npy",
    name == "dgcnn1.biases.npy",
    name == "dgcnn2.biases.npy",
    name == "dgcnn3.biases.npy",
    name == "dgcnn4.biases.npy",
    name == "agg.biases.npy"
  };

  bool isMatOpsIn1[] = {
    name == "transform_net1.tconv1.bn.transform_net1.tconv1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "transform_net1.tconv1.bn.transform_net1.tconv1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "transform_net1.tconv2.bn.transform_net1.tconv2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "transform_net1.tconv2.bn.transform_net1.tconv2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "transform_net1.tconv3.bn.transform_net1.tconv3.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "transform_net1.tconv3.bn.transform_net1.tconv3.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "transform_net1.tfc1.bn.transform_net1.tfc1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "transform_net1.tfc1.bn.transform_net1.tfc1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "transform_net1.tfc2.bn.transform_net1.tfc2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "transform_net1.tfc2.bn.transform_net1.tfc2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "dgcnn1.bn.dgcnn1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "dgcnn1.bn.dgcnn1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "dgcnn2.bn.dgcnn2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "dgcnn2.bn.dgcnn2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "dgcnn3.bn.dgcnn3.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "dgcnn3.bn.dgcnn3.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "dgcnn4.bn.dgcnn4.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "dgcnn4.bn.dgcnn4.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "agg.bn.agg.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "agg.bn.agg.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "fc1.bn.fc1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "fc1.bn.fc1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
    name == "fc2.bn.fc2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
    name == "fc2.bn.fc2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy"
  };

  bool isMatOpsIn2[] = {
    name == "transform_net1.tconv1.bn.gamma.npy",
    name == "transform_net1.tconv1.bn.beta.npy",
    name == "transform_net1.tconv2.bn.gamma.npy",
    name == "transform_net1.tconv2.bn.beta.npy",
    name == "transform_net1.tconv3.bn.gamma.npy",
    name == "transform_net1.tconv3.bn.beta.npy",
    name == "transform_net1.tfc1.bn.gamma.npy",
    name == "transform_net1.tfc1.bn.beta.npy",
    name == "transform_net1.tfc2.bn.gamma.npy",
    name == "transform_net1.tfc2.bn.beta.npy",
    name == "dgcnn1.bn.gamma.npy",
    name == "dgcnn1.bn.beta.npy",
    name == "dgcnn2.bn.gamma.npy",
    name == "dgcnn2.bn.beta.npy",
    name == "dgcnn3.bn.gamma.npy",
    name == "dgcnn3.bn.beta.npy",
    name == "dgcnn4.bn.gamma.npy",
    name == "dgcnn4.bn.beta.npy",
    name == "agg.bn.gamma.npy",
    name == "agg.bn.beta.npy",
    name == "fc1.bn.gamma.npy",
    name == "fc1.bn.beta.npy",
    name == "fc2.bn.gamma.npy",
    name == "fc2.bn.beta.npy",
    name == "transform_net1.tfc1.biases.npy",
    name == "transform_net1.tfc2.biases.npy",
    name == "fc1.biases.npy",
    name == "fc2.biases.npy",
    name == "fc3.biases.npy"
  };

  bool isMatMulIn2[] = {
    name == "transform_net1.tfc1.weights.npy",
    name == "transform_net1.tfc2.weights.npy",
    name == "fc1.weights.npy",
    name == "fc2.weights.npy",
    name == "fc3.weights.npy"
  };

  bool rsltConvWeight = false;
  bool rsltConvBias = false;
  bool rsltMatOpsIn1 = false;
  bool rsltMatOpsIn2 = false;
  bool rsltMatMulIn2 = false;

  for(bool item:isConvWeight){
    rsltConvWeight = rsltConvWeight | item;
  }
  for(bool item:isConvBias){
    rsltConvBias = rsltConvBias | item;
  }

  for(bool item:isMatOpsIn1){
    rsltMatOpsIn1 = rsltMatOpsIn1 | item;
  }
  for(bool item:isMatOpsIn2){
    rsltMatOpsIn2 = rsltMatOpsIn2 | item;
  }

  for(bool item:isMatMulIn2){
    rsltMatMulIn2 = rsltMatMulIn2 | item;
  }

  if(rsltConvWeight){
    return "conv_w";
  }
  if(rsltConvBias){
    return "conv_b";
  }

  if(rsltMatOpsIn1){
    return "matops_in1";
  }
  if(rsltMatOpsIn2){
    return "matops_in2";
  }

  if(rsltMatMulIn2){
    return "matmul_in2";
  }

  return "undefined_tag";
}
CWeightLoader::~CWeightLoader() {
  if(m_bIsLoaded){
    for(unsigned i=0;i<m_uWeightCount;i++){
      delete m_ptrWeightsCpu[i];
      delete m_ptrWeightsXil[i];
    }
    delete[](m_ptrWeightsCpu);
    delete[](m_ptrWeightsXil);
  }
}
