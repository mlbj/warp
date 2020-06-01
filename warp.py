import numpy as np

class Warp(object):
    def __init__(self,shape):
        # shape : (M,N) 
        # pars  : string com os parametros de warping
        # ii    : np.array de tamanho (M,N)
        # jj    : np.array de tamanho (M,N) 
        self.shape=shape
        self.pars='warp pars'
        self.ii=np.zeros(self.shape).astype(int)
        self.jj=np.zeros(self.shape).astype(int)
        
        
    def forward(self,f):
        # f : imagem de entrada, nao distorcida
        # g : imagem de saida, distorcida
        M,N=self.shape
        g=np.zeros(self.shape)
        ii=self.ii
        jj=self.jj
        for i in range(M):
            for j in range(N):
                if 0<=i+ii[i,j]<M and 0<=j+jj[i,j]<N:
                    g[i,j]=f[i+ii[i,j],j+jj[i,j]]
    
        #return g.astype('uint8')
        return g.astype(int)
     
    def backward(self,g):
        # g : imagem de saida, distorcida 
        # f : imagem de entrada, nao distorcida
        M,N=self.shape
        ii=self.ii
        jj=self.jj
        
        
        # + alternativamente, poderia inicia a reconstrucao com a imagem distorcida g
        # + futuramente mudar pra uma matriz flag com valores -1. 
        #   no final dessa função interpolar os valores que ainda
        #   possuem a flag.
        # + isso é uma alternativa para iniciar a matriz com zeros.
        #   se voce iniciar a matriz com zeros, no final vai ficar
        #   varias partes pretas na reconstrucao 
        # + aplicar alguma tecnica de inpainting tambem pode funcionar
        f=np.zeros(self.shape)
        
        #'''
        for i in range(M):
            for j in range(N):
                if 0<=i+ii[i,j]<M and 0<=j+jj[i,j]<N:
                    f[i+ii[i,j],j+jj[i,j]]=g[i,j]
        
        '''
        for i in range(M):
            for j in range(N):
                if 0<=i-ii[i,j]<M and 0<=j-jj[i,j]<N:
                    f[i,j]=g[i-ii[i,j],j-jj[i,j]]
        '''
        
        return f.astype(int)
    
class Butterworth(Warp):
    def __init__(self,shape=None,r=20,q=20,n=1,ii_flag=True,jj_flag=True):
        # r : raio de corte
        # q : quantidade de niveis
        # n : ordem da mascara butterworth 
        # saída em [0,1]
        
        if shape is not None:
            self.shape=shape
        
        M,N=self.shape
        v,u=np.indices((M,N)).astype(float)
        v=v-M/2
        u=u-N/2
        mask=(1/(np.sqrt(1+((u**2+v**2)/r**2)**n))).astype(float)
        mask=(mask*q)
        
        if ii_flag==True:
            self.ii=mask.astype(int)
        if jj_flag==True:
            self.jj=(mask.copy()).astype(int)
    
    
class Random(Warp):
    def __init__(self,shape=None,ii_flag=True,jj_flag=True,k=10):
        
        if shape is not None:
            self.shape=shape
        
        
        M,N=self.shape
        mask1=np.random.randint(0,k,size=self.shape)
        mask2=np.random.randint(0,k,size=self.shape)
        
        self.ii=mask1.astype(int)
        self.jj=mask2.astype(int)
   
