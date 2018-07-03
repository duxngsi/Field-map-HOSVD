import numpy as np

class DataSVDCompress:

    def __init__(self,data,keeps):
        #keep=[m,n,p] means reduce dimension of 1st, 2nd, 3rd order to m,n,p
        self.original_data=data
        self.original_shape=np.shape(data)
        self.order=np.size(self.original_shape)
        self.keeps=keeps
        #下面的等待函数填充
        self.v=list(np.ones(self.order))    #v是行向量！！!
        self.s=list(np.ones(self.order))
        self.compressed_data=data
        self.compressed_shape=list(self.original_shape)
        self.CoreT=0
        self.v0=list(np.ones(self.order))
        self.s0=list(np.ones(self.order))



    def __compress_nd1(self, which_dim=0, keep=10):
        # shape must be the same as data.shape
        # which_dim must smaller than order
        # keep must smaller than shape[which_dim]
        shape=self.compressed_shape
        switch_index = np.arange(0, self.order, step=1, dtype='int32')
        temp = switch_index[which_dim]
        switch_index[which_dim] = switch_index[self.order - 1]
        switch_index[self.order - 1] = temp
        # =========加工得到用于交换维度的数列
        data2 = np.transpose(self.compressed_data, switch_index)
        data2_flat = np.reshape(data2, (-1, shape[which_dim]))

        u, s, v = np.linalg.svd(data2_flat)
        su_flat = u[:,:keep] * s[:keep]
        switched_shape = np.take(shape, switch_index)  # 交换后的维度信息
        switched_shape[self.order - 1] = keep  # 交换并压缩后的维度信息
        su = np.reshape(su_flat, switched_shape)
        self.compressed_data = np.transpose(su, switch_index)  # 压缩处理的维度交换回原位置
        v = v[:keep, :]     #============与tf.svd不同，这里返回的v是行向量，被乘时不用转置
        print(which_dim)
        self.v[which_dim]=v
        self.s[which_dim]=s

        self.compressed_shape[which_dim]=keep

    def __easy_svd_1d(self,which_dim,keep):
        # some times high dim makes to large index
        shape2=self.compressed_shape
        switch_index = np.arange(0, self.order, step=1, dtype='int32')
        temp = switch_index[which_dim]
        switch_index[which_dim] = switch_index[self.order - 1]
        switch_index[self.order - 1] = temp
        # =========加工得到用于交换维度的数列
        data2 = np.transpose(self.compressed_data, switch_index)
        most_inner_index= np.size(self.original_shape)-1
        data2T=np.swapaxes(data2, most_inner_index, most_inner_index-1)
        cmats=np.matmul(data2T,data2)
        bb = np.arange(most_inner_index - 1)
        bb = tuple(bb)
        cmat=np.average(cmats,axis=bb)
        u,s,v=np.linalg.svd(cmat)
        v = v[:keep, :]     #
        print(which_dim)
        self.v[which_dim]=v
        self.s[which_dim]=s
        self.compressed_shape[which_dim] = keep
        data3=np.tensordot(data2,v.T,axes=1)
        self.compressed_data=np.transpose(data3,switch_index)

    def compress_ndm(self):

        for i in range(self.order):
            if self.keeps[i]==self.original_shape[i]:
                self.v[i]=1
            else:
                #self.__compress_nd1(i,self.keeps[i])
                self.__easy_svd_1d(i,self.keeps[i])

    def HOSVD(self):
        for i in range(self.order):
            self.__easy_svd_1d(i,self.original_shape[i])


    def __decompress_nd1(self,which_dim):

        switch_index = np.arange(0, self.order, step=1, dtype='int32')
        temp = switch_index[which_dim]
        switch_index[which_dim] = switch_index[self.order - 1]
        switch_index[self.order - 1] = temp

        data1=np.transpose(self.recovered_data,switch_index)    #将待解压维度换到内部
        v=self.v[which_dim]
        result=np.tensordot(data1, v,axes=1)
        self.recovered_data=np.transpose(result,switch_index)   #转换回原位置

    def recover(self):
        self.recovered_data=self.compressed_data
        for i in range(self.order):
            if self.keeps[i]!=self.original_shape[i]:
                self.__decompress_nd1(i)

