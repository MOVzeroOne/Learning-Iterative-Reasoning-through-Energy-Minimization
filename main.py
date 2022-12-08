import jax.numpy as jnp 
from jax import lax,random,jit,grad, value_and_grad,vmap
from functools import partial 
import matplotlib.pyplot as plt 
import flax 
import flax.linen as nn 
from typing import Sequence 
from collections import deque 
import optax 
from tqdm import tqdm 
"""
in this case we just have 1 input. (not the input output pair to learn a relation between)
So we need to learn to update that input so it satisfies a target.
(this single output is a 2d position so we can visualize it nicely)

energy based model version of a unimodal generator producing a single output

run the algorithm for n steps and safe the output so it can be used as an intermediate target later on.
make sure to vmap into the network, as we don't want to have the average across batches.

sample from the buffer and sample from the data distribution.


"""

class network(nn.Module):
    features : Sequence[int] 
    @nn.compact
    def __call__(self,x):
        for i,feature in enumerate(self.features):
            if(i != len(self.features)-1):
                x = nn.swish(nn.Dense(feature)(x))
            else:
                x = nn.Dense(feature)(x)
        
        return jnp.sum(x)  #inner loss

def wrapper(param,pos,pos_buffer,targets,module,max_depth):
    @jit
    @partial(grad,has_aux=True)
    def outer(param,pos,pos_buffer):
        @jit
        @partial(vmap,in_axes=(0,None)) 
        @grad
        def pos_grads(pos,param): 
            return module.apply(param,pos)
        
        def loop(loop_num,pos,param):
            #updates of the pos
            return pos - pos_grads(pos,param)


        pos = lax.fori_loop(lower=0,upper=max_depth,body_fun=partial(loop,param=param),init_val=pos)
        pos_buffer = lax.fori_loop(lower=0,upper=max_depth,body_fun=partial(loop,param=param),init_val=pos_buffer)

        batch_size = pos.shape[0]
        return jnp.sum(optax.l2_loss(pos,targets))/batch_size + jnp.sum(optax.l2_loss(pos_buffer,targets))/batch_size,lax.stop_gradient(pos)
    return outer(param,pos,pos_buffer)

def visualize(param,pos,module,target,time_steps=50):
    @jit
    @grad
    def pos_grads(pos,param):
        return module.apply(param,pos)

    n = 100
    x = jnp.linspace(-10,10,n)
    y = jnp.linspace(-10,10,n)
    X,Y = jnp.meshgrid(x,y)
    
    X = jnp.resize(X,(n**2,1))
    Y = jnp.resize(Y,(n**2,1))
    
    pos_grid = jnp.concatenate((X,Y),axis=1)

    for _ in tqdm(range(time_steps),ascii=True):
        energy_grid = pos_grads(pos_grid,param) #((grad_x,grad_y),(grad_x,grad_y))

        
        grads = pos_grads(pos,param)
        pos = pos- grads
        plt.cla()
        plt.ylim(ymax=10,ymin=-10)
        plt.xlim(xmax=10,xmin=-10)
        plt.rcParams['image.cmap'] = 'tab20'
        plt.quiver(pos_grid[:,0],pos_grid[:,1],-energy_grid[:,0],-energy_grid[:,1],jnp.arctan2(-energy_grid[:,0],-energy_grid[:,1]),minlength=0.001)
        plt.scatter(pos[:,0],pos[:,1])
        plt.scatter(target[:,0],target[:,1],color="red")
        plt.pause(0.001)

class buffer(deque):
    def __init__(self,buffer_size):
        super().__init__(maxlen=buffer_size)
    
    def sample(self,key,batch_size,axis=0):
        indices = random.choice(key,jnp.arange(len(self)),(batch_size,))
        return jnp.concatenate([self[index] for index in indices],axis=axis)



if __name__ == "__main__":
    BATCH_SIZE = 100
    DEPTH = 40
    BUFFER_SIZE = 10000
    #targets = jnp.array([[5.0,5.0],[-2.0,-2.0]])
    targets = jnp.array([[5.0,5.0]])
    
    key = random.PRNGKey(0)
    key_pos,key_param,key = random.split(key,3)
    pos = random.uniform(key_pos,(BATCH_SIZE,2))
    module = network([100,100,100,1])
    param = module.init(key_param,pos)

    optimizer = optax.adam(learning_rate=0.001)

    opt_state = optimizer.init(param)

    buff = buffer(BUFFER_SIZE)

    for i in tqdm(range(1000),ascii=True):
        #settingup targets, inputs and buffered inputs 
        key_pos,key_buff,key_choice,key = random.split(key,4)
        pos = random.uniform(key_pos,(BATCH_SIZE,2))*10 -5
        if(len(buff) == 0):
            pos_buff = jnp.copy(pos) 
        else:
            pos_buff = buff.sample(key_buff,BATCH_SIZE)
        target =random.choice(key_choice,targets,(BATCH_SIZE,))
       
        #calculate the run and update the network
        grads,new_buffer_pos = wrapper(param,pos,pos_buff,target,module,DEPTH) 
        updates, opt_state = optimizer.update(grads, opt_state, param)
        param = optax.apply_updates(param, updates)

        #update the buffer 
        for new_poss in new_buffer_pos:
            buff.append(jnp.expand_dims(new_poss,0))

        #visualization
        if(i % 1 ==0):
            visualize(param,pos,module,target)