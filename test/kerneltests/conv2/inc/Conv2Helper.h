//
// Created by saleh on 3/5/20.
//

#pragma once

template< typename T>
void Conv2Kernel1x1CPU(
        const T* inputData,
        const T* weightData,
        const T* biasData,
        T* outputData,
        const unsigned shape_b,
        unsigned shape_n,
        unsigned shape_k,
        unsigned shape_d,
        unsigned shape_chOut){
    unsigned int indxS1,indxS2,indxD;

    for(int b=0;b<shape_b;b++){
        for(int n=0;n<shape_n;n++){
            for(int k=0;k<shape_k;k++){
                indxS1= b*shape_n*shape_k*shape_d +
                        n*shape_k*shape_d +
                        k*shape_d +
                        0;
                for(int ch=0;ch<shape_chOut;ch++){
                    float sum=0;
                    for(int d=0;d<shape_d;d++){
                        indxS2 = d*shape_chOut + ch;
                        sum += inputData[indxS1+d] * weightData[indxS2];
                    }
                    indxD = b*shape_n*shape_k*shape_chOut +
                            n*shape_k*shape_chOut +
                            k*shape_chOut +
                            ch;
                    outputData[indxD] = sum + biasData[ch];
                }
            }
        }
    }
}
