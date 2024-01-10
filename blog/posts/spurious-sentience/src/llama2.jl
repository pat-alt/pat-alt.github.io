# Adapted from https://juliazoid.com/walking-through-text-generation-with-llama-2-using-transformers-jl-5735f8dffe9e

using Flux
using HuggingFaceApi
using Metal 
using StatsBase
using StringViews
using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace

textenc = hgf"meta-llama/Llama-2-7b-chat-hf:tokenizer"
model = todevice(hgf"meta-llama/Llama-2-7b-chat-hf:ForCausalLM")