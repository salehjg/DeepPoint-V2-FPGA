#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "models/CModel1.h"

CModel1::CModel1(
    PLATFORMS targetPlatform,
    unsigned datasetOffset,
    unsigned batchSize,
    unsigned pointsPerPointCloud,
    unsigned knnK,
    bool useShapeNetInstead,
    bool enableOclProfiling,
    bool enableMemBankCrossing,
    bool enableCpuUtilization,
    bool enableTensorDumps){

  m_bUseShapeNet = useShapeNetInstead;
  m_uClassCount = m_bUseShapeNet?55:40;
  m_uDatasetOffset = datasetOffset;
  m_uBatchSize = batchSize;
  m_uPointsPerCloud = pointsPerPointCloud;
  m_uKnnK = knnK;
  m_eTargetPlatform = targetPlatform;
  m_ptrPlatSelection = new CPlatformSelection(
      targetPlatform,
      m_bUseShapeNet,
      true,
      enableOclProfiling,
      enableMemBankCrossing,
      enableCpuUtilization,
      enableTensorDumps
  );
}

CModel1::~CModel1() {
  delete(m_ptrPlatSelection);
}

void CModel1::SetDatasetData(std::string &pathNumpyData) {
  m_oNumpyObjectData = cnpy::npy_load(pathNumpyData);
  auto rawNpyShape = m_oNumpyObjectData.shape;
  auto rawNpyRank = rawNpyShape.size();
  ConditionCheck(rawNpyRank==3, "The input numpy files for the dataset do not have a rank of 3.");
  ConditionCheck(rawNpyShape[0]>=m_uBatchSize, "The input numpy files for the dataset have less models than the target batch-size.");
  ConditionCheck(rawNpyShape[1]==m_uPointsPerCloud, "The input numpy files for the dataset should have the set number of points per model.");
  ConditionCheck(rawNpyShape[2]==3, "The input numpy files for the dataset should have 3 features per point.");
  ConditionCheck(m_uDatasetOffset+m_uBatchSize<=rawNpyShape[0], "The input numpy files for the dataset are too small for the current dataset offset.");
  unsigned offset = m_uDatasetOffset*(m_uPointsPerCloud*3);
  auto *ptrBuff = m_oNumpyObjectData.data<float>() + offset;
  m_ptrDatasetDataTn = CTensorBasePtr(
      new CTensor<float>({m_uBatchSize,m_uPointsPerCloud,3}, ptrBuff));
}

void CModel1::SetDatasetLabels(std::string &pathNumpyLabels) {
  // dataType of npy file should be int32, NOT uchar8!
  // use dataset_B5_labels_int32.npy
  m_oNumpyObjectLabels = cnpy::npy_load(pathNumpyLabels);
  auto rawNpyShape = m_oNumpyObjectLabels.shape;
  auto rawNpyRank = rawNpyShape.size();
  ConditionCheck(rawNpyRank==2 && rawNpyShape[1]==1, "The input numpy files for the dataset labels do not have a rank of 2 with shape[1]=1.");
  ConditionCheck(m_uDatasetOffset+m_uBatchSize<=rawNpyShape[0], "The input numpy files for the dataset labels are too small for the current dataset offset.");
  unsigned offset = m_uDatasetOffset;
  auto *_ptrBuff = m_oNumpyObjectLabels.data<int>() + offset;
  auto *ptrBuff = reinterpret_cast<unsigned*>(_ptrBuff);
  m_ptrDatasetLabelsTn = CTensorBasePtr(
      new CTensor<unsigned>({m_uBatchSize}, ptrBuff));
}

CTensorBasePtr CModel1::GetDataTn() {
  return m_ptrDatasetDataTn;
}

CTensorBasePtr CModel1::GetLabelTn() {
  return m_ptrDatasetLabelsTn;
}

unsigned CModel1::GetBatchSize() {
  return m_uBatchSize;
}

unsigned CModel1::GetClassCount() {
  return m_uClassCount;
}

PLATFORMS CModel1::GetTargetPlatform() {
  return m_eTargetPlatform;
}

CTensorBasePtr CModel1::FullyConnectedForward(CTensorBasePtr inputTn, CTensorBasePtr weightsTn, CTensorBasePtr biasesTn) {
  auto tmp = m_ptrPlatSelection->MatMul(GetTargetPlatform(), inputTn, weightsTn);
  return m_ptrPlatSelection->BasicOps(GetTargetPlatform(), tmp, biasesTn, BASIC_OPS::ADD);
}

CTensorBasePtr CModel1::BatchNormForward(CTensorBasePtr inputTn,
                                       CTensorBasePtr gammaTn,
                                       CTensorBasePtr betaTn,
                                       CTensorBasePtr emaAveTn,
                                       CTensorBasePtr emaVarTn) {
 
  const float bn_decay = 0.5f;
  const auto rank = inputTn->GetRank();
  
  CTensorBasePtr mu;
  CTensorBasePtr var;
  
  if(rank==4){
    //mu and var is of shape (dim3)
    mu = m_ptrPlatSelection->Mean(GetTargetPlatform(), inputTn, {1,1,1,0});
    var = m_ptrPlatSelection->Variance(GetTargetPlatform(), inputTn, {1,1,1,0});

    // Exponential Moving Average for mu and var
    CTensorBasePtr update_delta_ave, update_delta_var;
    CTensorBasePtr update_delta_ave2, update_delta_var2;

    update_delta_ave = m_ptrPlatSelection->BasicOps(GetTargetPlatform(), emaAveTn, mu, BASIC_OPS::SUB);
    update_delta_ave2 = m_ptrPlatSelection->BasicOps(GetTargetPlatform(), update_delta_ave, bn_decay, BASIC_OPS::MUL_ELEMENTWISE);
    update_delta_var = m_ptrPlatSelection->BasicOps(GetTargetPlatform(), emaVarTn, var, BASIC_OPS::SUB);
    update_delta_var2 = m_ptrPlatSelection->BasicOps(GetTargetPlatform(), update_delta_var, bn_decay, BASIC_OPS::MUL_ELEMENTWISE);

    auto final_ave =  m_ptrPlatSelection->BasicOps(GetTargetPlatform(), emaAveTn, update_delta_ave2, BASIC_OPS::SUB);
    auto final_var =  m_ptrPlatSelection->BasicOps(GetTargetPlatform(), emaVarTn, update_delta_var2, BASIC_OPS::SUB);
    auto xNormTmp1 = m_ptrPlatSelection->BasicOps(GetTargetPlatform(),inputTn,final_ave, BASIC_OPS::SUB);
    auto xNormTmp2 = m_ptrPlatSelection->BasicOps(GetTargetPlatform(),final_var,1e-8, BASIC_OPS::ADD);
    auto xNormTmp3 = m_ptrPlatSelection->Sqrt(GetTargetPlatform(),xNormTmp2);
    auto xNorm = m_ptrPlatSelection->BasicOps(GetTargetPlatform(),xNormTmp1,xNormTmp3, BASIC_OPS::DIV_ELEMENTWISE);
    auto rsltTmp1 = m_ptrPlatSelection->BasicOps(GetTargetPlatform(),xNorm,gammaTn, BASIC_OPS::MUL_ELEMENTWISE);
    auto rsltTn = m_ptrPlatSelection->BasicOps(GetTargetPlatform(),rsltTmp1,betaTn, BASIC_OPS::ADD);
    
    return rsltTn;
  } else if(rank==2){
    //mu and var is of shape (dim1)
    mu = m_ptrPlatSelection->Mean(GetTargetPlatform(), inputTn, {1,0});
    var = m_ptrPlatSelection->Variance(GetTargetPlatform(), inputTn, {1,0});

    // Exponential Moving Average for mu and var
    CTensorBasePtr update_delta_ave, update_delta_var;
    CTensorBasePtr update_delta_ave2, update_delta_var2;

    update_delta_ave = m_ptrPlatSelection->BasicOps(GetTargetPlatform(), emaAveTn, mu, BASIC_OPS::SUB);
    update_delta_ave2 = m_ptrPlatSelection->BasicOps(GetTargetPlatform(), update_delta_ave, bn_decay, BASIC_OPS::MUL_ELEMENTWISE);
    update_delta_var = m_ptrPlatSelection->BasicOps(GetTargetPlatform(), emaVarTn, var, BASIC_OPS::SUB);
    update_delta_var2 = m_ptrPlatSelection->BasicOps(GetTargetPlatform(), update_delta_var, bn_decay, BASIC_OPS::MUL_ELEMENTWISE);

    auto final_ave =  m_ptrPlatSelection->BasicOps(GetTargetPlatform(), emaAveTn, update_delta_ave2, BASIC_OPS::SUB);
    auto final_var =  m_ptrPlatSelection->BasicOps(GetTargetPlatform(), emaVarTn, update_delta_var2, BASIC_OPS::SUB);
    auto xNormTmp1 = m_ptrPlatSelection->BasicOps(GetTargetPlatform(),inputTn,final_ave, BASIC_OPS::SUB);
    auto xNormTmp2 = m_ptrPlatSelection->BasicOps(GetTargetPlatform(),final_var,1e-8, BASIC_OPS::ADD);
    auto xNormTmp3 = m_ptrPlatSelection->Sqrt(GetTargetPlatform(),xNormTmp2);
    auto xNorm = m_ptrPlatSelection->BasicOps(GetTargetPlatform(),xNormTmp1,xNormTmp3, BASIC_OPS::DIV_ELEMENTWISE);
    auto rsltTmp1 = m_ptrPlatSelection->BasicOps(GetTargetPlatform(),xNorm,gammaTn, BASIC_OPS::MUL_ELEMENTWISE);
    auto rsltTn = m_ptrPlatSelection->BasicOps(GetTargetPlatform(),rsltTmp1,betaTn, BASIC_OPS::ADD);
    
    return rsltTn;
  }else{
    ConditionCheck(false, "Something has gone wrong.")
  }
}

CTensorBasePtr CModel1::GetEdgeFeatures(CTensorBasePtr inputTn, CTensorBasePtr knnTn) {
  //Gather TopK's indices from the input array.
  auto point_cloud_neighbors = m_ptrPlatSelection->Gather(GetTargetPlatform(),inputTn,knnTn,1);

  //BxNxD into BxNxKxD
  //inputTn->ExpandDims(2);
  auto point_cloud_central = m_ptrPlatSelection->Tile(GetTargetPlatform(),inputTn,2,m_uKnnK);
  //inputTn->SqueezeDims();

  auto features = m_ptrPlatSelection->BasicOps(GetTargetPlatform(), point_cloud_neighbors,point_cloud_central,BASIC_OPS::SUB);

  //concatenate centrals and features (BxNxKxD) and (BxNxKxD)
  auto edge_feature = m_ptrPlatSelection->Concat2(GetTargetPlatform(), point_cloud_central,features,3);
  
  return edge_feature;
}

CTensorBasePtr CModel1::PairwiseDistance(CTensorBasePtr inputTn) {
  auto point_cloud_transpose = m_ptrPlatSelection->Transpose(GetTargetPlatform(),inputTn);
  auto point_cloud_inner =  m_ptrPlatSelection->MatMul(GetTargetPlatform(),inputTn,point_cloud_transpose);
  auto point_cloud_inner2 = m_ptrPlatSelection->BasicOps(GetTargetPlatform(),point_cloud_inner,-2.0f,BASIC_OPS::MUL_ELEMENTWISE);
  auto point_cloud_inner2p2 = m_ptrPlatSelection->Square(GetTargetPlatform(),inputTn);

  ///TODO: REDUCE LAYER IS CHANGED, CHECK IT.
  //auto point_cloud_sum = m_ptrPlatSelection->ReduceSum(GetTargetPlatform(),point_cloud_inner2p2,false,false,true);
  auto point_cloud_sum = m_ptrPlatSelection->Reduce(GetTargetPlatform(),point_cloud_inner2p2,REDUCTION_OPS::SUM,1,{0,0,1});

  auto point_cloud_sum_tiled =  m_ptrPlatSelection->Tile(GetTargetPlatform(),point_cloud_sum,2,m_uPointsPerCloud); //The result is BxNxK for k=N
  auto point_cloud_sum_transpose_tiled =  m_ptrPlatSelection->Tile(GetTargetPlatform(),point_cloud_sum,1,m_uPointsPerCloud); //The result is BxkxN for k=N
  auto rsltTmpTn = m_ptrPlatSelection->BasicOps(GetTargetPlatform(),point_cloud_sum_tiled,point_cloud_sum_transpose_tiled, BASIC_OPS::ADD); //both input tensors are BxNxN
  auto rsltTn = m_ptrPlatSelection->BasicOps(GetTargetPlatform(),rsltTmpTn,point_cloud_inner2, BASIC_OPS::ADD); //both input tensors are BxNxN

  return rsltTn;
}

CTensorBasePtr CModel1::TransformNet(CTensorBasePtr edgeFeaturesTn) {

  CTensorBasePtr net;
  {
    auto net1 = m_ptrPlatSelection->Conv2D(
        GetTargetPlatform(),
        edgeFeaturesTn,
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv1.weights.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv1.biases.npy")
    );

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A01_tnet_conv.npy",net1);

    auto net2 = BatchNormForward(
        net1,
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv1.bn.gamma.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv1.bn.beta.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv1.bn.transform_net1.tconv1.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv1.bn.transform_net1.tconv1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
    );

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A02_tnet_bn.npy",net2);

    auto net3 = m_ptrPlatSelection->ReLU(GetTargetPlatform(),net2);

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A03_tnet_relu.npy",net3);

    net = net3;
  }

  //----------------------------------------------------------------------------
  {
    auto net1 = m_ptrPlatSelection->Conv2D(
        GetTargetPlatform(),
        net,
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv2.weights.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv2.biases.npy")
    );

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A04_tnet_conv.npy",net1);

    auto net2 = BatchNormForward(
        net1,
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv2.bn.gamma.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv2.bn.beta.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv2.bn.transform_net1.tconv2.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv2.bn.transform_net1.tconv2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
    );

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A05_tnet_bn.npy",net2);

    auto net3 = m_ptrPlatSelection->ReLU(GetTargetPlatform(),net2);

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A06_tnet_relu.npy",net3);

    net = net3;
  }

  //----------------------------------------------------------------------------
  {
    ///TODO: CHECK THIS. REDUCE LAYER HAS BEEN CHANGED.
    //auto net1 = m_ptrPlatSelection->ReduceMax(GetTargetPlatform(),net,2);
    auto net1 = m_ptrPlatSelection->Reduce(GetTargetPlatform(),net,REDUCTION_OPS::MAX,1,{0,0,1,0});

    net1->ExpandDims(2);
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A07_tnet_pool.npy",net1);
    net = net1;
  }

  //----------------------------------------------------------------------------
  {
    auto net1 = m_ptrPlatSelection->Conv2D(
        GetTargetPlatform(),
        net,
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv3.weights.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv3.biases.npy")
    );

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A08_tnet_conv.npy",net1);

    auto net2 = BatchNormForward(
        net1,
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv3.bn.gamma.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv3.bn.beta.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv3.bn.transform_net1.tconv3.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tconv3.bn.transform_net1.tconv3.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
    );

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A09_tnet_bn.npy",net2);

    auto net3 = m_ptrPlatSelection->ReLU(GetTargetPlatform(),net2);

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A10_tnet_relu.npy",net3);
    net = net3;
  }

  //----------------------------------------------------------------------------
  {
    ///TODO: CHECK THIS, REDUCE LAYER HAS BEEN CHANGED.
    //auto net1 = m_ptrPlatSelection->ReduceMax(GetTargetPlatform(),net,1);
    auto net1 = m_ptrPlatSelection->Reduce(GetTargetPlatform(),net,REDUCTION_OPS::MAX,1,{0,1,0,0});

    net1->SqueezeDims();
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A11_tnet_pool.npy",net1);
    net = net1;
  }

  //----------------------------------------------------------------------------
  //FC
  // net is Bx1024
  {
    auto net1 = FullyConnectedForward(
        net,
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tfc1.weights.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tfc1.biases.npy")
    );

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A12_tnet_fc.npy",net1);

    auto net2 = BatchNormForward(
        net1,
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tfc1.bn.gamma.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tfc1.bn.beta.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tfc1.bn.transform_net1.tfc1.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tfc1.bn.transform_net1.tfc1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
    );

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A13_tnet_bn.npy",net2);

    auto net3 = m_ptrPlatSelection->ReLU(GetTargetPlatform(),net2);

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A13_tnet_relu.npy",net3);

    net = net3;
  }

  //----------------------------------------------------------------------------
  //FC
  // net is Bx1024
  {
    auto net1 = FullyConnectedForward(
        net,
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tfc2.weights.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tfc2.biases.npy")
    );

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A14_tnet_fc.npy",net1);

    auto net2 = BatchNormForward(
        net1,
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tfc2.bn.gamma.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tfc2.bn.beta.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tfc2.bn.transform_net1.tfc2.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
        m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
            GetTargetPlatform(),"transform_net1.tfc2.bn.transform_net1.tfc2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
    );

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A15_tnet_bn.npy",net2);

    auto net3 = m_ptrPlatSelection->ReLU(GetTargetPlatform(),net2);

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A16_tnet_relu.npy",net3);

    net = net3;
  }

  //----------------------------------------------------------------------------
  {
    auto weights = m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
        GetTargetPlatform(),"transform_net1.transform_XYZ.weights.npy");
    auto _biases = m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
        GetTargetPlatform(),"transform_net1.transform_XYZ.biases.npy");

    float eyeData[] = {1,0,0,
                       0,1,0,
                       0,0,1};
    CTensorBasePtr eye(new CTensor<float>({9}, eyeData));

    auto biases = m_ptrPlatSelection->BasicOps(GetTargetPlatform(),_biases,eye, BASIC_OPS::ADD);
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A17_biass_added.npy",biases);

    auto transformTn = m_ptrPlatSelection->MatMul(GetTargetPlatform(),net,weights);
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A18_transform_batch.npy",transformTn);


    auto transformFinalTn = m_ptrPlatSelection->BasicOps(GetTargetPlatform(), transformTn, biases, BASIC_OPS::ADD);
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"A19_transform_batch_bias.npy",transformFinalTn);

    // Forcibly use the CPU tensor. This will allow us to use reshape without worrying about the padded last dim policy.
    return m_ptrPlatSelection->CrossThePlatformIfNeeded(PLATFORMS::CPU, transformFinalTn); // Force CPU
  }
}

CTensorBasePtr CModel1::Execute() {
  CTensorBasePtr net;
  CTensorBasePtr net_BxNx3;

  CTensorBasePtr endpoint_0, endpoint_1, endpoint_2, endpoint_3;

  //----------------------------------------------------------------------------------------
  SPDLOG_LOGGER_INFO(logger,"Starting Process...");
  SPDLOG_LOGGER_INFO(logger,"Batch Size: {}", m_uBatchSize);
  SPDLOG_LOGGER_INFO(logger,"Point Count: {}", m_uPointsPerCloud);

  //----------------------------------------------------------------------------------------
  // TransferNet(net_BxNx3 is this layer's input)
  {
    net_BxNx3 = GetDataTn();
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B00_input_pcl_BxNxD.npy",net_BxNx3);

    auto adj_matrix = PairwiseDistance(net_BxNx3);
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B01_tnet_adj_matrix.npy",adj_matrix);

    auto nn_idx = m_ptrPlatSelection->TopK(GetTargetPlatform(),adj_matrix,2,m_uKnnK);
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B02_tnet_nn_idx.npy",nn_idx);
    
    auto edge_features = GetEdgeFeatures(net_BxNx3, nn_idx);
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B03_tnet_edgef.npy",edge_features);

    auto transform = TransformNet(edge_features);
    transform->Reshape({m_uBatchSize,3,3});
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B04_tnet_3x3.npy",transform);

    net = m_ptrPlatSelection->MatMul(GetTargetPlatform(),  net_BxNx3, transform);
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"C01_pcl.npy",net);
  }

  //----------------------------------------------------------------------------------------
  // DGCNN Layer #0
  SPDLOG_LOGGER_INFO(logger,"DGCCN0 Started...");
  {
    auto adj_matrix = PairwiseDistance(net);
    auto nn_idx = m_ptrPlatSelection->TopK(GetTargetPlatform(),adj_matrix,2,m_uKnnK);
    auto edge_features = GetEdgeFeatures(net,nn_idx);
    auto net1 = m_ptrPlatSelection->Conv2D(GetTargetPlatform(),
                                             edge_features,
                                             m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                                 GetTargetPlatform(),"dgcnn1.weights.npy"),
                                             m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                                 GetTargetPlatform(),"dgcnn1.biases.npy")
    );
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"C02_dg1_conv.npy",net1);

    auto net2 = BatchNormForward(net1,
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn1.bn.gamma.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn1.bn.beta.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn1.bn.dgcnn1.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn1.bn.dgcnn1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
    );
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"C03_dg1_bn.npy",net2);

    auto net3 = m_ptrPlatSelection->ReLU(GetTargetPlatform(),net2);

    ///TODO: CHECK THIS! REDUCE LAYER HAS BEEN CHANGED!
    //auto net4 = m_ptrPlatSelection->ReduceMax(GetTargetPlatform(),net3,2);
    auto net4 = m_ptrPlatSelection->Reduce(GetTargetPlatform(),net3,REDUCTION_OPS::MAX,1,{0,0,1,0});

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B05_dg1_pool.npy",net4);

    net = net4;
    endpoint_0 = net;
  }

  //----------------------------------------------------------------------------------------
  // DGCNN Layer #1
  SPDLOG_LOGGER_INFO(logger,"DGCCN1 Started...");
  {
    auto adj_matrix = PairwiseDistance(net);
    auto nn_idx = m_ptrPlatSelection->TopK(GetTargetPlatform(),adj_matrix,2,m_uKnnK);
    auto edge_features = GetEdgeFeatures(net,nn_idx);
    auto net1 = m_ptrPlatSelection->Conv2D(GetTargetPlatform(),
                                             edge_features,
                                             m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                                 GetTargetPlatform(),"dgcnn2.weights.npy"),
                                             m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                                 GetTargetPlatform(),"dgcnn2.biases.npy")
    );

    auto net2 = BatchNormForward(net1,
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn2.bn.gamma.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn2.bn.beta.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn2.bn.dgcnn2.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn2.bn.dgcnn2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
    );

    auto net3 = m_ptrPlatSelection->ReLU(GetTargetPlatform(),net2);

    ///TODO: CHECK THIS, REDUCE LAYER HAS BEEN CHANGED!
    //auto net4 = m_ptrPlatSelection->ReduceMax(GetTargetPlatform(),net3,2);
    auto net4 = m_ptrPlatSelection->Reduce(GetTargetPlatform(),net3,REDUCTION_OPS::MAX,1,{0,0,1,0});

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B06_dg2_pool.npy",net4);

    net = net4;
    endpoint_1 = net;
  }

  //----------------------------------------------------------------------------------------
  // DGCNN Layer #2
  SPDLOG_LOGGER_INFO(logger,"DGCCN2 Started...");
  {
    auto adj_matrix = PairwiseDistance(net);
    auto nn_idx = m_ptrPlatSelection->TopK(GetTargetPlatform(), adj_matrix,2,m_uKnnK);
    auto edge_features = GetEdgeFeatures(net,nn_idx);
    auto net1 = m_ptrPlatSelection->Conv2D(GetTargetPlatform(),
                                             edge_features,
                                             m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                                 GetTargetPlatform(),"dgcnn3.weights.npy"),
                                             m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                                 GetTargetPlatform(),"dgcnn3.biases.npy")
    );

    auto net2 = BatchNormForward(net1,
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn3.bn.gamma.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn3.bn.beta.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn3.bn.dgcnn3.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn3.bn.dgcnn3.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
    );

    auto net3 = m_ptrPlatSelection->ReLU(GetTargetPlatform(),net2);

    ///TODO: CHECK THIS, REDUCE LAYER HAS BEEN CHANGED!
    //auto net4 = m_ptrPlatSelection->ReduceMax(GetTargetPlatform(),net3,2);
    auto net4 = m_ptrPlatSelection->Reduce(GetTargetPlatform(),net3,REDUCTION_OPS::MAX,1,{0,0,1,0});

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B07_dg3_pool.npy",net4);

    net = net4;
    endpoint_2 = net;
  }

  //----------------------------------------------------------------------------------------
  // DGCNN Layer #3
  SPDLOG_LOGGER_INFO(logger,"DGCCN3 Started...");
  {
    auto adj_matrix = PairwiseDistance(net);
    auto nn_idx = m_ptrPlatSelection->TopK(GetTargetPlatform(),adj_matrix,2,m_uKnnK);
    auto edge_features = GetEdgeFeatures(net,nn_idx);
    auto net1 = m_ptrPlatSelection->Conv2D(GetTargetPlatform(),
                                             edge_features,
                                             m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                                 GetTargetPlatform(),"dgcnn4.weights.npy"),
                                             m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                                 GetTargetPlatform(),"dgcnn4.biases.npy")
    );

    auto net2 = BatchNormForward(net1,
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn4.bn.gamma.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn4.bn.beta.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn4.bn.dgcnn4.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"dgcnn4.bn.dgcnn4.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
    );

    auto net3 = m_ptrPlatSelection->ReLU(GetTargetPlatform(),net2);

    ///TODO: CHECK THIS, REDUCE LAYER HAS BEEN CHANGED.
    //auto net4 = m_ptrPlatSelection->ReduceMax(GetTargetPlatform(),net3,2);
    auto net4 = m_ptrPlatSelection->Reduce(GetTargetPlatform(),net3,REDUCTION_OPS::MAX,1,{0,0,1,0});

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B08_dg4_pool.npy",net4);

    net = net4;
    endpoint_3 = net;
  }

  //----------------------------------------------------------------------------------------
  SPDLOG_LOGGER_INFO(logger,"Agg Layer Started...");
  {
    endpoint_0->ExpandDims(2);
    endpoint_1->ExpandDims(2);
    endpoint_2->ExpandDims(2);
    endpoint_3->ExpandDims(2);
    auto concatA = m_ptrPlatSelection->Concat2(GetTargetPlatform(), endpoint_0, endpoint_1, 3);
    auto concatB = m_ptrPlatSelection->Concat2(GetTargetPlatform(), concatA, endpoint_2, 3);
    auto concatC = m_ptrPlatSelection->Concat2(GetTargetPlatform(), concatB, endpoint_3, 3);
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B09_agg_concat.npy",concatC);

    // DIM2(m_uKnnK) of the concatenated tensor is ONE, NOT 'm_uKnnK'
    auto net1 = m_ptrPlatSelection->Conv2D(GetTargetPlatform(),
                                             concatC,
                                             m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                                 GetTargetPlatform(),"agg.weights.npy"),
                                             m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                                 GetTargetPlatform(),"agg.biases.npy")
    );

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B10_agg_conv.npy",net1);

    auto net2 = BatchNormForward(net1,
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"agg.bn.gamma.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"agg.bn.beta.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"agg.bn.agg.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"agg.bn.agg.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
    );


    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B11_agg_bn.npy",net2);
    auto net3 = m_ptrPlatSelection->ReLU(GetTargetPlatform(),net2);

    ///TODO: CHECK THIS, REDUCE LAYER HAS BEEN CHANGED.
    //auto net4 = m_ptrPlatSelection->ReduceMax(GetTargetPlatform(),net3,1);
    auto net4 = m_ptrPlatSelection->Reduce(GetTargetPlatform(),net3,REDUCTION_OPS::MAX,1,{0,1,0,0});

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B12_agg_pool.npy",net4);

    net = net4;
  }

  //----------------------------------------------------------------------------------------
  //RESHAPING TO (Bx-1)
  {
    net->SqueezeDims();
  }

  //----------------------------------------------------------------------------------------
  //FC1
  //net is of shape Bx1x1x1024
  SPDLOG_LOGGER_INFO(logger,"FC Layer1 Started...");
  {
    auto net1 = FullyConnectedForward(net,
                                           m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                               GetTargetPlatform(),"fc1.weights.npy"),
                                           m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                               GetTargetPlatform(),"fc1.biases.npy")
    );
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B13_fc.npy",net1);

    //net1 is of shape Bx1x1x512
    auto net2 = BatchNormForward(net1,
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"fc1.bn.gamma.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"fc1.bn.beta.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"fc1.bn.fc1.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"fc1.bn.fc1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
    );
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B14_fc.npy",net2);

    auto net3 = m_ptrPlatSelection->ReLU(GetTargetPlatform(),net2);
    net = net3;
  }

  //----------------------------------------------------------------------------------------
  //FC2
  //net is of shape Bx1x1x512
  SPDLOG_LOGGER_INFO(logger,"FC Layer2 Started...");
  {
    auto net1 = FullyConnectedForward(net,
                                           m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                               GetTargetPlatform(),"fc2.weights.npy"),
                                           m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                               GetTargetPlatform(),"fc2.biases.npy")
    );
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B15_fc.npy",net1);

    //net1 is of shape Bx1x1x512
    auto net2 = BatchNormForward(net1,
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"fc2.bn.gamma.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"fc2.bn.beta.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"fc2.bn.fc2.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                                      m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                          GetTargetPlatform(),"fc2.bn.fc2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
    );

    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B16_fc.npy",net2);

    auto net3 = m_ptrPlatSelection->ReLU(GetTargetPlatform(),net2);
    net = net3;
  }

  //----------------------------------------------------------------------------------------
  //FC3
  //net is of shape Bx1x1x256
  SPDLOG_LOGGER_INFO(logger,"FC Layer3 Started...");
  {
    auto net1 = FullyConnectedForward(net,
                                           m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                               GetTargetPlatform(),"fc3.weights.npy"),
                                           m_ptrPlatSelection->GetClassPtrWeightLoader()->AccessWeights(
                                               GetTargetPlatform(),"fc3.biases.npy")
    );
    m_ptrPlatSelection->DumpToNumpyFile(PLATFORMS::CPU,"B17_fc.npy",net1);
    net = net1;
  }

  //----------------------------------------------------------------------------------------
  //force output tensor platform to be CPU
  return m_ptrPlatSelection->CrossThePlatformIfNeeded(PLATFORMS::CPU, net);
}
