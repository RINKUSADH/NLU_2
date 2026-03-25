import torch
import torch.nn as nn
import torch.optim as optim
import random
import string
import math
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- TASK 0: Dataset ---
with open("TrainingNames.txt", "r", encoding="utf-8") as f:
    names = [line.strip().lower() for line in f.readlines()]

all_chars = sorted(list(set("".join(names))))
all_chars.insert(0, "<pad>")
all_chars.append("<eos>")

vocab_size = len(all_chars)
char2idx = {ch: i for i, ch in enumerate(all_chars)}
idx2char = {i: ch for i, ch in enumerate(all_chars)}
PAD_IDX = char2idx["<pad>"]

# --- TASK 1A: MANUAL BACKWARD AUTOGRAD FUNCTIONS ---

class CustomLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.reshape(-1, grad_output.size(-1)).t().matmul(input.reshape(-1, input.size(-1)))
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)
                
        return grad_input, grad_weight, grad_bias

class CustomRNNStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_t, h_prev, W_ih, W_hh):
        # Forward Math: h_t = tanh(x_t * W_ih^T + h_prev * W_hh^T)
        z_t = x_t.matmul(W_ih.t()) + h_prev.matmul(W_hh.t())
        h_t = torch.tanh(z_t)
        ctx.save_for_backward(x_t, h_prev, W_ih, W_hh, h_t)
        return h_t

    @staticmethod
    def backward(ctx, grad_h_t):
        x_t, h_prev, W_ih, W_hh, h_t = ctx.saved_tensors
        
        # Derivative of tanh: (1 - tanh^2(x))
        grad_z_t = grad_h_t * (1.0 - h_t ** 2)
        
        # Chain rule back to inputs and weights
        grad_x_t = grad_z_t.matmul(W_ih)
        grad_h_prev = grad_z_t.matmul(W_hh)
        grad_W_ih = grad_z_t.t().matmul(x_t)
        grad_W_hh = grad_z_t.t().matmul(h_prev)
        
        return grad_x_t, grad_h_prev, grad_W_ih, grad_W_hh

class CustomLSTMStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_t, h_prev, c_prev, W_ih, W_hh):
        # Calculate all 4 gates simultaneously
        gates = x_t.matmul(W_ih.t()) + h_prev.matmul(W_hh.t())
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
        
        i_t = torch.sigmoid(i_gate)
        f_t = torch.sigmoid(f_gate)
        g_t = torch.tanh(g_gate)
        o_t = torch.sigmoid(o_gate)
        
        # Cell and Hidden State updates
        c_t = f_t * c_prev + i_t * g_t
        tanh_c_t = torch.tanh(c_t)
        h_t = o_t * tanh_c_t
        
        ctx.save_for_backward(x_t, h_prev, c_prev, W_ih, W_hh, i_t, f_t, g_t, o_t, c_t, tanh_c_t)
        return h_t, c_t

    @staticmethod
    def backward(ctx, grad_h_t, grad_c_t):
        x_t, h_prev, c_prev, W_ih, W_hh, i_t, f_t, g_t, o_t, c_t, tanh_c_t = ctx.saved_tensors
        
        # 1. Gradients from h_t to output gate and cell state
        grad_o_t = grad_h_t * tanh_c_t
        grad_c_t_internal = grad_h_t * o_t * (1.0 - tanh_c_t ** 2) + grad_c_t
        
        # 2. Gradients through cell state to other gates
        grad_f_t = grad_c_t_internal * c_prev
        grad_c_prev = grad_c_t_internal * f_t
        grad_i_t = grad_c_t_internal * g_t
        grad_g_t = grad_c_t_internal * i_t
        
        # 3. Derivatives of activation functions (sigmoid & tanh)
        di = grad_i_t * i_t * (1.0 - i_t)
        df = grad_f_t * f_t * (1.0 - f_t)
        dg = grad_g_t * (1.0 - g_t ** 2)
        do = grad_o_t * o_t * (1.0 - o_t)
        
        # 4. Reconstruct gate gradients and pass back to weights
        dgates = torch.cat([di, df, dg, do], dim=1)
        
        grad_x_t = dgates.matmul(W_ih)
        grad_h_prev = dgates.matmul(W_hh)
        grad_W_ih = dgates.t().matmul(x_t)
        grad_W_hh = dgates.t().matmul(h_prev)
        
        return grad_x_t, grad_h_prev, grad_c_prev, grad_W_ih, grad_W_hh


# --- TASK 1B: MODEL IMPLEMENTATION (FROM SCRATCH) ---

def init_weights(m):
    # Standard Xavier initialization to prevent vanishing gradients
    if isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_IDX)
        
        # Learnable Parameters replacing nn.RNN and nn.Linear
        self.W_ih = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        self.W_fc = nn.Parameter(torch.Tensor(vocab_size, hidden_size))
        self.b_fc = nn.Parameter(torch.Tensor(vocab_size))
        
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)
        nn.init.xavier_uniform_(self.W_fc)
        nn.init.zeros_(self.b_fc)
    
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        batch_size, seq_len, _ = embedded.size()
        
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            # Uses our custom manual forward/backward function
            h_t = CustomRNNStep.apply(x_t, h_t, self.W_ih, self.W_hh)
            outputs.append(h_t.unsqueeze(1))
            
        outputs = torch.cat(outputs, dim=1)
        
        batch_idx = torch.arange(batch_size, device=x.device)
        last_out = outputs[batch_idx, lengths - 1, :]
        
        # Uses custom manual linear backward
        return CustomLinear.apply(last_out, self.W_fc, self.b_fc)

class BLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.half_hidden = hidden_size // 2
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_IDX)
        
        # Forward LSTM parameters
        self.W_ih_f = nn.Parameter(torch.Tensor(4 * self.half_hidden, hidden_size))
        self.W_hh_f = nn.Parameter(torch.Tensor(4 * self.half_hidden, self.half_hidden))
        
        # Backward LSTM parameters
        self.W_ih_b = nn.Parameter(torch.Tensor(4 * self.half_hidden, hidden_size))
        self.W_hh_b = nn.Parameter(torch.Tensor(4 * self.half_hidden, self.half_hidden))
        
        self.W_fc = nn.Parameter(torch.Tensor(vocab_size, hidden_size))
        self.b_fc = nn.Parameter(torch.Tensor(vocab_size))
        
        for p in [self.W_ih_f, self.W_hh_f, self.W_ih_b, self.W_hh_b, self.W_fc]:
            nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.b_fc)
    
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        batch_size, seq_len, _ = embedded.size()
        
        # Forward Pass
        h_f = torch.zeros(batch_size, self.half_hidden, device=x.device)
        c_f = torch.zeros(batch_size, self.half_hidden, device=x.device)
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h_f, c_f = CustomLSTMStep.apply(x_t, h_f, c_f, self.W_ih_f, self.W_hh_f)
            
        # Backward Pass
        h_b = torch.zeros(batch_size, self.half_hidden, device=x.device)
        c_b = torch.zeros(batch_size, self.half_hidden, device=x.device)
        
        for t in range(seq_len - 1, -1, -1):
            x_t = embedded[:, t, :]
            h_b, c_b = CustomLSTMStep.apply(x_t, h_b, c_b, self.W_ih_b, self.W_hh_b)
            
        hidden_cat = torch.cat((h_f, h_b), dim=1)
        return CustomLinear.apply(hidden_cat, self.W_fc, self.b_fc)

class RNNAttention(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(RNNAttention, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_IDX)
        
        self.W_ih = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        self.W_attn = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_attn = nn.Parameter(torch.Tensor(1))
        
        self.W_fc = nn.Parameter(torch.Tensor(vocab_size, hidden_size))
        self.b_fc = nn.Parameter(torch.Tensor(vocab_size))
        
        for p in [self.W_ih, self.W_hh, self.W_attn, self.W_fc]:
            nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.b_fc)
        nn.init.zeros_(self.b_attn)
    
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        batch_size, seq_len, _ = embedded.size()
        
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h_t = CustomRNNStep.apply(x_t, h_t, self.W_ih, self.W_hh)
            outputs.append(h_t.unsqueeze(1))
            
        output = torch.cat(outputs, dim=1)
        
        mask = (x != PAD_IDX).unsqueeze(-1).float()
        
        # Attention scoring: Linear transform of hidden states
        attn_scores = CustomLinear.apply(output.reshape(-1, self.hidden_size), self.W_attn, self.b_attn)
        attn_scores = attn_scores.reshape(batch_size, seq_len, 1)
        
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * output, dim=1)
        
        return CustomLinear.apply(context, self.W_fc, self.b_fc)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

hidden_size = 128
models = {
    'VanillaRNN': VanillaRNN(vocab_size, hidden_size).to(device),
    'BLSTM': BLSTM(vocab_size, hidden_size).to(device),
    'RNNAttention': RNNAttention(vocab_size, hidden_size).to(device)
}

print(f"Vanilla RNN parameters: {count_parameters(models['VanillaRNN'])}")
print(f"BLSTM parameters: {count_parameters(models['BLSTM'])}")
print(f"RNN+Attention parameters: {count_parameters(models['RNNAttention'])}")

# Training loop
X_train = []
y_train = []

for name in names:
    chars = list(name) + ["<eos>"]
    for i in range(1, len(chars)):
        prefix = [char2idx[ch] for ch in chars[:i]]
        target = char2idx[chars[i]]
        X_train.append(torch.tensor(prefix))
        y_train.append(target)

epochs = 15
loss_fn = nn.CrossEntropyLoss()
batch_size = 1024

for name, model in models.items():
    print(f"\nTraining {name}...")
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    model.train()
    
    indices = list(range(len(X_train)))
    
    for ep in range(epochs):
        random.shuffle(indices)
        total_loss = 0
        
        for k in range(0, len(X_train), batch_size):
            batch_idx = indices[k:k+batch_size]
            batch_x_list = [X_train[i] for i in batch_idx]
            batch_y = torch.tensor([y_train[i] for i in batch_idx]).to(device)
            lengths = torch.tensor([len(x) for x in batch_x_list]).to(device)
            
            batch_x_padded = pad_sequence(batch_x_list, batch_first=True, padding_value=PAD_IDX).to(device)
            
            optimizer.zero_grad()
            out = model(batch_x_padded, lengths)
            loss = loss_fn(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)
            
        print(f"  Epoch {ep+1} Loss: {total_loss / len(X_train):.4f}")

# Generation
print("\n--- Generating and Evaluating ---")
def generate_name(model, start_char, max_len=20):
    model.eval()
    with torch.no_grad():
        current_prefix = [char2idx[start_char]]
        for _ in range(max_len):
            x = torch.tensor([current_prefix]).to(device)
            lengths = torch.tensor([len(current_prefix)]).to(device)
            out = model(x, lengths)
            probs = torch.softmax(out[0] / 0.8, dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            
            if next_idx == char2idx["<eos>"] or next_idx == char2idx["<pad>"]:
                break
            current_prefix.append(next_idx)
            
        return "".join([idx2char[i] for i in current_prefix])

training_names_set = set(names)

with open('report_generation.txt', 'w') as f_report:
    for name, model in models.items():
        generated_names = []
        for _ in range(300):
            start_c = random.choice(list(string.ascii_lowercase))
            if start_c not in char2idx:
                start_c = 'a'
            gen = generate_name(model, start_c)
            if len(gen) > 2:
                generated_names.append(gen)
                
        unique_generated = set(generated_names)
        novel_names = unique_generated - training_names_set
        
        diversity = len(unique_generated) / max(1, len(generated_names))
        novelty = len(novel_names) / max(1, len(unique_generated))
        
        report_str = f"Model: {name}\n"
        report_str += f"  Novelty Rate: {novelty*100:.2f}%\n"
        report_str += f"  Diversity: {diversity*100:.2f}%\n"
        report_str += f"  Sample Names:\n"
        for n in list(unique_generated)[:10]:
            report_str += f"    - {n.capitalize()}\n"
        report_str += "\n"
        
        print(report_str.strip())
        f_report.write(report_str)
        
        with open(f"GeneratedNames_{name}.txt", "w") as f:
            for n in unique_generated:
                f.write(n.capitalize() + "\n")

print("\nProblem 2 completed successfully.")