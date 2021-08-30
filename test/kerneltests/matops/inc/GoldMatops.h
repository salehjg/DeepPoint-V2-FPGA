#pragma once

template <typename T>
void GoldMatops(
        const T* inputA,
        const T* inputB,
        T* output,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dim3,
        const unsigned dim0B,
        const unsigned dim1B,
        const unsigned dim2B,
        const unsigned dim3B,
        const unsigned rankA,
        const unsigned rankB,
        const int mode){

    unsigned indxS1, indxS2;
    unsigned dim0B_IsNotZero, dim1B_IsNotZero, dim2B_IsNotZero, dim3B_IsNotZero;

    unsigned tmp =15>>(4-rankB);
    dim0B_IsNotZero = (tmp >> 3) & 1;
    dim1B_IsNotZero = (tmp >> 2) & 1;
    dim2B_IsNotZero = (tmp >> 1) & 1;
    dim3B_IsNotZero = (tmp >> 0) & 1;

    if(rankB==1 && dim0B==0&&dim1B==0&&dim2B==0&&dim3B==1){//scalar value
        dim3B_IsNotZero=0; //force it to be zero, so in the kernel, indxS2 would be zero;
    }

    for(int d0=0;d0<dim0;d0++){
        for(int d1=0;d1<dim1;d1++) {
            for(int d2=0;d2<dim2;d2++) {
                for(int d3=0;d3<dim3;d3++) {
                    indxS1 = d0*dim1*dim2*dim3+
                             d1*dim2*dim3+
                             d2*dim3+
                             d3;
                    indxS2 = d0 * dim1B * dim2B * dim3B * dim0B_IsNotZero +
                             d1 * dim2B * dim3B * dim1B_IsNotZero +
                             d2 * dim3B * dim2B_IsNotZero +
                             d3 * dim3B_IsNotZero;

                    if(mode==0)//Add
                        output[indxS1] = inputA[indxS1] + inputB[indxS2];
                    else if(mode==1)//Sub
                        output[indxS1] = inputA[indxS1] - inputB[indxS2];
                    else if(mode==2)//Mul (element wise)
                        output[indxS1] = inputA[indxS1] * inputB[indxS2];
                    else if(mode==3)//Div (element wise)
                        output[indxS1] = inputA[indxS1] / inputB[indxS2];
                }
            }
        }
    }
}
