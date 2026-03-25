import numpy as np

def relu(x):       return np.maximum(0.0, x)
def relu_grad(x):  return (x > 0.0).astype(np.float64)

def sigmoid(x):
    pos = x >= 0
    result = np.empty_like(x, dtype=np.float64)
    result[pos]  = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    result[~pos] = exp_x / (1.0 + exp_x)
    return result

def normalise_adjacency(edge_index, n_nodes, edge_weight=None):
    A = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    if edge_weight is not None:
        w = np.clip(edge_weight, -3.0, 3.0).astype(np.float64)
        A[edge_index[0], edge_index[1]] = w
    else:
        A[edge_index[0], edge_index[1]] = 1.0
    A_tilde = A + np.eye(n_nodes)
    d = A_tilde.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-12)))
    return d_inv_sqrt @ A_tilde @ d_inv_sqrt

def layer_norm(x, eps=1e-6):
    mu  = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mu) / (std + eps)

class GCNLayer:
    def __init__(self, in_dim, out_dim, seed=0):
        rng    = np.random.default_rng(seed)
        self.W = rng.standard_normal((in_dim, out_dim)) * np.sqrt(1.0 / in_dim)
        self.b = np.zeros(out_dim)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
    def forward(self, A_hat, H):
        self._A_hat = A_hat; self._H_in = H
        self._Z = layer_norm(A_hat @ H @ self.W + self.b)
        self._H_out = relu(self._Z)
        return self._H_out
    def backward(self, dH_out):
        dZ = dH_out * relu_grad(self._Z)
        n = np.linalg.norm(dZ)
        if n > 1.0: dZ = dZ / n
        self._dW = self._H_in.T @ (self._A_hat.T @ dZ)
        self._db = dZ.sum(axis=0)
        return self._A_hat @ (dZ @ self.W.T)
    def adam_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8):
        for p, g, m, v in [(self.W,self._dW,self.mW,self.vW),(self.b,self._db,self.mb,self.vb)]:
            m[:]=b1*m+(1-b1)*g; v[:]=b2*v+(1-b2)*g**2
            p -= lr*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+eps)

class DenseLayer:
    def __init__(self, in_dim, out_dim, seed=0):
        rng    = np.random.default_rng(seed)
        self.W = rng.standard_normal((in_dim, out_dim)) * np.sqrt(1.0 / in_dim)
        self.b = np.zeros(out_dim)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
    def forward(self, x): self._x = x; return x @ self.W + self.b
    def backward(self, d): self._dW = self._x.T @ d; self._db = d.sum(0); return d @ self.W.T
    def adam_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8):
        for p, g, m, v in [(self.W,self._dW,self.mW,self.vW),(self.b,self._db,self.mb,self.vb)]:
            m[:]=b1*m+(1-b1)*g; v[:]=b2*v+(1-b2)*g**2
            p -= lr*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+eps)

class BrainGCN:
    def __init__(self, in_dim=5, hidden_dim=32, seed=42):
        self.gcn1=GCNLayer(in_dim,hidden_dim,seed); self.gcn2=GCNLayer(hidden_dim,hidden_dim,seed+1)
        self.dense=DenseLayer(hidden_dim,1,seed+2); self._t=0
    def forward(self, graph):
        A_hat = normalise_adjacency(graph["edge_index"], graph["x"].shape[0], graph["edge_weight"])
        H1=self.gcn1.forward(A_hat,graph["x"]); H2=self.gcn2.forward(A_hat,H1)
        pool=H2.mean(axis=0,keepdims=True); logit=self.dense.forward(pool)
        prob=float(sigmoid(logit).squeeze())
        if np.isnan(prob): prob=0.5
        self._H2=H2; self._prob=prob; return prob
    def backward(self, y_true, lr=1e-3):
        self._t+=1; y=float(y_true); p=self._prob; eps=1e-9
        d_logit=np.array([[p-y]]); d_pool=self.dense.backward(d_logit)
        N=self._H2.shape[0]; d_H2=np.repeat(d_pool,N,axis=0)/N
        d_H1=self.gcn2.backward(d_H2); self.gcn1.backward(d_H1)
        for l in [self.gcn1,self.gcn2,self.dense]: l.adam_step(lr,self._t)
        return float(-(y*np.log(p+eps)+(1-y)*np.log(1-p+eps)))
    def predict_proba(self, graph): return self.forward(graph)
    def predict(self, graph, threshold=0.5): return int(self.predict_proba(graph)>=threshold)
