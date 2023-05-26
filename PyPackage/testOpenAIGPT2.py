from Seq2SeqSharp import TensorUtils, ProcessorTypeEnums, WeightTensor, WeightTensorFactory, ComputeGraphTensor
import torch
import numpy as np
from System import Array
import Seq2SeqSharp
import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tu = TensorUtils()
tu.InitDevices(ProcessorTypeEnums.CPU, [0]);

graph = ComputeGraphTensor(WeightTensorFactory(), 0, False)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2")

inputText = "Seq2SeqSharp is a C# based deep learning toolkit."
generated = tokenizer.encode(inputText)
context = torch.tensor([generated])
past_key_values = None

for i in range(100):

    outputs = model(context, past_key_values=past_key_values)
    past_key_values = outputs.past_key_values
    last_hidden_states = outputs.logits.squeeze()

    if len(last_hidden_states.size()) == 1:
        last_hidden_states = last_hidden_states.unsqueeze(0)

    tensor = WeightTensor(last_hidden_states.size(), 0, 0)
    ll = np.array(last_hidden_states.cpu().detach().type(torch.FloatTensor)).ravel()
    tensor.SetWeightArray(Seq2SeqSharp.asNetArray(ll))
    tensor = graph.Softmax(tensor)
    tensor = graph.TopPSample(tensor, 0.7)
    resultArray = tensor.ToWeightArray()
    token = torch.tensor(int(resultArray[resultArray.Length - 1]))

    context = token.unsqueeze(0)
    generated += [token.tolist()]

sequence = tokenizer.decode(generated)
print(sequence)
